// --- DOM Elements ---
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const imageUpload = document.getElementById('imageUpload');
const nameInput = document.getElementById('nameInput');
const relationshipInput = document.getElementById('relationshipInput'); // New input
const registerButton = document.getElementById('registerButton');
const registerStatus = document.getElementById('registerStatus');
const registeredList = document.getElementById('registeredList');
const clearAllButton = document.getElementById('clearAllButton');
const clearStatus = document.getElementById('clearStatus');
const loader = document.getElementById('loader');

// --- Configuration ---
const MODEL_URL = './models'; // Path to face-api.js models
const FACE_MATCHER_THRESHOLD = 0.55; // Stricter matching (lower value = stricter)
const STABILIZATION_FRAMES = 8; // Show last known label for up to X frames if current is unknown
const BOX_PROXIMITY_THRESHOLD = 50; // Max pixel distance between box centers to be considered the "same" face for stabilization
const LABEL_SMOOTHING_FACTOR = 0.3; // Smoothing factor for label position (0-1, lower = smoother but slower)

// --- Label Styling ---
const TEXT_PADDING = 8; // Increased padding
const TEXT_MARGIN = 15; // Increased margin from face box
const FONT_SIZE = 15; // Slightly smaller font
const BACKGROUND_COLOR = 'rgba(0, 0, 0, 0.55)'; // Slightly less opaque background
const FONT_COLOR = 'white'; // Text color
const FONT_STYLE = `normal ${FONT_SIZE}px 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif`; // Changed font
const BACKGROUND_RADIUS = 6; // Slightly larger radius

// --- State Variables ---
let labeledFaceDescriptors = [];
let faceMatcher = null;
let modelsLoaded = false;
let recognitionInterval = null;
// trackedFaces stores info about faces being tracked across frames for stabilization and smoothing
// Structure: { trackingId: { label: 'Name|Rel', framesUnknown: 0, box: {...}, smoothedPos: {x, y} } }
let trackedFaces = {};


// --- 1. Load Models ---
async function loadModels() {
    console.log("Loading models from:", MODEL_URL);
    loader.style.display = 'block'; // Show loader
    loader.innerText = "Loading Models...";
    try {
        // Load all necessary models concurrently
        await Promise.all([
            faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL), // Fast detector for general use
            faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL), // Detects facial landmarks
            faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL) // Computes face descriptors for recognition
        ]);
        console.log("Models loaded successfully");
        modelsLoaded = true;
        loader.style.display = 'none'; // Hide loader
        loadRegisteredFaces(); // Load saved faces after models are ready
        startVideo();          // Start webcam after models are ready
    } catch (error) {
        console.error("Error loading models:", error);
        loader.innerText = "Error loading models. Check console.";
        loader.style.display = 'block'; // Keep loader visible on error
    }
}

// --- 2. Start Webcam ---
function startVideo() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("getUserMedia is not supported in this browser.");
        loader.innerText = "Webcam access not supported by this browser.";
        loader.style.display = 'block';
        return;
    }

    navigator.mediaDevices.getUserMedia({ video: {} })
        .then(stream => {
            video.srcObject = stream;
            video.addEventListener('loadedmetadata', () => {
                console.log("Video metadata loaded.");
            });
             console.log("Webcam access granted.");
        })
        .catch(err => {
            console.error("Error accessing webcam:", err);
            loader.innerText = "Error accessing webcam. Please grant permission.";
            loader.style.display = 'block';
        });
}

// --- 3. Handle Registration ---
registerButton.addEventListener('click', async () => {
    if (!modelsLoaded) {
        registerStatus.textContent = 'Models not loaded yet.';
        registerStatus.style.color = 'red';
        return;
    }

    const name = nameInput.value.trim();
    const relationship = relationshipInput.value.trim(); // Get relationship
    const imageFile = imageUpload.files[0];
    const combinedLabel = `${name}|${relationship}`; // Combine for storage

    // Validation
    if (!name || !relationship || !imageFile) {
        registerStatus.textContent = 'Please provide name, relationship, and image.';
        registerStatus.style.color = 'red';
        return;
    }
    if (labeledFaceDescriptors.some(desc => desc.label.split('|')[0] === name)) {
        registerStatus.textContent = `Name "${name}" seems already registered. Use a different name or clear existing entries if needed.`;
        registerStatus.style.color = 'orange';
        // return; // Allow duplicates for now, user can manage via list/clear
    }


    registerStatus.textContent = 'Processing image...';
    registerStatus.style.color = 'orange';

    try {
        const image = await faceapi.bufferToImage(imageFile);
        registerStatus.textContent = 'Detecting face...';

        // Detect face, landmarks, and compute descriptor
        const detection = await faceapi.detectSingleFace(image, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 })) // Use same detector as live
                                     .withFaceLandmarks()
                                     .withFaceDescriptor();

        if (!detection) {
            registerStatus.textContent = 'No face detected in the uploaded image.';
            registerStatus.style.color = 'red';
            return;
        }
        if (!detection.descriptor) {
             registerStatus.textContent = 'Could not compute face descriptor.';
             registerStatus.style.color = 'red';
             return;
        }

        // Create and add the labeled descriptor
        const descriptor = new faceapi.LabeledFaceDescriptors(combinedLabel, [detection.descriptor]);
        labeledFaceDescriptors.push(descriptor);

        saveRegisteredFaces();    // Save to localStorage
        updateFaceMatcher();      // Update the matcher
        updateRegisteredList();   // Update the displayed list

        registerStatus.textContent = `Registered ${name} (${relationship}) successfully!`;
        registerStatus.style.color = 'green';
        nameInput.value = ''; // Clear inputs
        relationshipInput.value = '';
        imageUpload.value = ''; // Clear file input

    } catch (error) {
        console.error("Registration error:", error);
        registerStatus.textContent = 'Registration failed. See console for details.';
        registerStatus.style.color = 'red';
    }
});

// --- 4. Face Detection and Recognition on Video ---
video.addEventListener('play', () => {
    if (!modelsLoaded) {
        console.warn("Attempted to start detection before models loaded.");
        return; // Don't start if models aren't ready
    }
    console.log("Video playing, starting detection interval.");

    // Create canvas context
    const ctx = canvas.getContext('2d');

    // Function to setup/resize canvas
    const setupCanvas = () => {
        if (video.videoWidth === 0 || video.videoHeight === 0 || video.clientWidth === 0 || video.clientHeight === 0) {
            console.log("Video dimensions not ready yet for canvas setup.");
            return null;
        }
        const displaySize = { width: video.clientWidth, height: video.clientHeight };
        faceapi.matchDimensions(canvas, displaySize);
        console.log("Canvas dimensions matched to video display:", displaySize);
        return displaySize;
    };

    let displaySize = setupCanvas();

    // Handle resize events
    window.addEventListener('resize', () => {
        displaySize = setupCanvas();
        trackedFaces = {}; // Clear tracked faces on resize
    });

    // Clear previous interval if any
    if (recognitionInterval) {
        clearInterval(recognitionInterval);
    }

    // Start the detection loop
    recognitionInterval = setInterval(async () => {
        if (video.paused || video.ended || !modelsLoaded || !displaySize) {
            if (!displaySize) displaySize = setupCanvas(); // Try setup again
            if (!displaySize) return; // Skip frame if canvas setup still not possible
        }

        // Perform detection
        const detections = await faceapi.detectAllFaces(video, new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 }))
                                       .withFaceLandmarks()
                                       .withFaceDescriptors();

        // Resize results
        const resizedDetections = faceapi.resizeResults(detections, displaySize);

        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const currentTrackedFaces = {}; // Store faces recognized/tracked in *this* frame

        if (faceMatcher && resizedDetections.length > 0) {
            const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor));

            results.forEach((result, i) => {
                const box = resizedDetections[i].detection.box;
                let label = result.label; // "Name|Relationship" or "unknown"
                let framesUnknown = 0;
                let trackingId = null; // ID for linking across frames
                let currentSmoothedPos = null; // Smoothed position for drawing

                // --- Stabilization Logic ---
                let closestTrackedFace = null;
                let minDistance = Infinity;
                let closestTrackedId = null;

                for (const id in trackedFaces) {
                    const trackedFace = trackedFaces[id];
                    const distance = getCenterDistance(box, trackedFace.box);
                    if (distance < BOX_PROXIMITY_THRESHOLD && distance < minDistance) {
                        minDistance = distance;
                        closestTrackedFace = trackedFace;
                        closestTrackedId = id;
                    }
                }

                if (label === 'unknown') {
                    if (closestTrackedFace && closestTrackedFace.framesUnknown < STABILIZATION_FRAMES) {
                        label = closestTrackedFace.label;
                        framesUnknown = closestTrackedFace.framesUnknown + 1;
                        trackingId = closestTrackedId;
                        // Keep the smoothed position from the tracked face
                        currentSmoothedPos = closestTrackedFace.smoothedPos;
                        // console.log(`Stabilization: Using previous label '${label.split('|')[0]}' for unknown face (ID: ${trackingId})`);
                    } else {
                        trackingId = null; // Don't track unknowns unless stabilized
                    }
                } else {
                    // Recognized face
                    framesUnknown = 0;
                    trackingId = closestTrackedId || `face_${Date.now()}_${i}`; // Assign new ID if not matched

                    // --- Position Smoothing ---
                    // Calculate the target position for the label (top-right of the box)
                    const targetPos = {
                        x: box.topRight.x + TEXT_MARGIN,
                        y: box.y + (box.height / 2) // Vertically centered target
                    };

                    // Get the previous smoothed position, or initialize if new face
                    const previousSmoothedPos = trackedFaces[trackingId]?.smoothedPos || targetPos;

                    // Apply linear interpolation (lerp) for smooth movement
                    currentSmoothedPos = {
                        x: previousSmoothedPos.x + (targetPos.x - previousSmoothedPos.x) * LABEL_SMOOTHING_FACTOR,
                        y: previousSmoothedPos.y + (targetPos.y - previousSmoothedPos.y) * LABEL_SMOOTHING_FACTOR
                    };
                }
                // --- End Stabilization & Smoothing Logic ---

                // Only draw and track if the label is known and we have a position
                if (label !== 'unknown' && trackingId && currentSmoothedPos) {
                    const [name, relationship] = label.split('|');
                    // Draw name and relationship using the smoothed position
                    drawNameAndRelationship(ctx, currentSmoothedPos, name, relationship);

                    // Track this face for the next frame
                    currentTrackedFaces[trackingId] = {
                        label: label,
                        framesUnknown: framesUnknown,
                        box: box, // Store the raw box for distance calculation next frame
                        smoothedPos: currentSmoothedPos // Store the smoothed position
                    };
                }
            });
        }

        // Update tracked faces for the next iteration
        trackedFaces = currentTrackedFaces;

    }, 100); // Interval in ms

    // Cleanup on pause/end
    video.addEventListener('pause', () => {
        if (recognitionInterval) clearInterval(recognitionInterval);
        console.log("Video paused, detection stopped.");
        if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
        trackedFaces = {};
    });
    video.addEventListener('ended', () => {
        if (recognitionInterval) clearInterval(recognitionInterval);
        console.log("Video ended, detection stopped.");
        if (ctx && canvas) ctx.clearRect(0, 0, canvas.width, canvas.height);
        trackedFaces = {};
    });
});

// --- 5. Custom Drawing Function ---
// Now accepts smoothedPos {x, y} instead of the raw box for positioning
function drawNameAndRelationship(ctx, smoothedPos, name, relationship) {
    const text = `${name} (${relationship})`;

    // Set font style for measurement
    ctx.font = FONT_STYLE;
    const textMetrics = ctx.measureText(text);
    const textWidth = textMetrics.width;
    const textHeight = FONT_SIZE; // Approximate height

    // Calculate position for the background rectangle based on smoothedPos
    // smoothedPos.x is the target left edge of the background rect
    // smoothedPos.y is the target vertical center
    const rectX = smoothedPos.x;
    const rectY = smoothedPos.y - (textHeight / 2) - TEXT_PADDING; // Adjust Y to be top edge
    const rectWidth = textWidth + (2 * TEXT_PADDING);
    const rectHeight = textHeight + (2 * TEXT_PADDING);

    // --- Draw Background ---
    ctx.fillStyle = BACKGROUND_COLOR;
    drawRoundedRect(ctx, rectX, rectY, rectWidth, rectHeight, BACKGROUND_RADIUS);
    ctx.fill();

    // --- Draw Text ---
    ctx.font = FONT_STYLE;
    ctx.fillStyle = FONT_COLOR;
    ctx.shadowColor = 'transparent'; // Ensure no shadow
    ctx.shadowBlur = 0;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;

    // Calculate text position (centered within the background rect)
    const textX = rectX + TEXT_PADDING;
    ctx.textBaseline = 'middle'; // Align text vertically center
    const textY = rectY + rectHeight / 2; // Center Y position within the rect

    ctx.fillText(text, textX, textY);

    // Reset baseline
    ctx.textBaseline = 'alphabetic';
}

// --- 6. Persistence using localStorage ---
function saveRegisteredFaces() {
    if (labeledFaceDescriptors.length === 0) {
        localStorage.removeItem('faceRecognitionData');
        console.log("No faces to save, cleared localStorage.");
        return;
    }
    try {
        const serializableData = labeledFaceDescriptors.map(lfd => ({
            label: lfd.label,
            descriptors: lfd.descriptors.map(d => Array.from(d))
        }));
        localStorage.setItem('faceRecognitionData', JSON.stringify(serializableData));
        console.log("Faces saved to localStorage.");
    } catch (error) {
        console.error("Error saving face data to localStorage:", error);
    }
}

function loadRegisteredFaces() {
    const savedData = localStorage.getItem('faceRecognitionData');
    if (savedData) {
        try {
            const parsedData = JSON.parse(savedData);
            labeledFaceDescriptors = parsedData.map(item =>
                new faceapi.LabeledFaceDescriptors(
                    item.label,
                    item.descriptors.map(d => new Float32Array(d))
                )
            );
            console.log("Loaded faces from localStorage:", labeledFaceDescriptors.map(lfd => lfd.label));
        } catch (error) {
            console.error("Failed to parse saved face data:", error);
            localStorage.removeItem('faceRecognitionData');
            labeledFaceDescriptors = [];
        }
    } else {
        console.log("No saved faces found in localStorage.");
        labeledFaceDescriptors = [];
    }
    updateFaceMatcher();
    updateRegisteredList();
}

// --- 7. Update Face Matcher ---
function updateFaceMatcher() {
    if (labeledFaceDescriptors.length > 0) {
        try {
            faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, FACE_MATCHER_THRESHOLD);
            console.log("FaceMatcher updated with labels:", labeledFaceDescriptors.map(lfd => lfd.label));
        } catch(error) {
             console.error("Error creating FaceMatcher:", error);
             faceMatcher = null;
        }
    } else {
        faceMatcher = null;
        console.log("No registered faces, FaceMatcher is null.");
    }
}

// --- 8. Update UI List ---
function updateRegisteredList() {
    registeredList.innerHTML = ''; // Clear current list
    if (labeledFaceDescriptors.length === 0) {
        registeredList.innerHTML = '<li>No faces registered yet.</li>';
    } else {
        labeledFaceDescriptors.forEach((lfd, index) => {
            const li = document.createElement('li');
            const [name, relationship] = lfd.label.split('|');
            li.textContent = `${name} (${relationship || 'N/A'})`;

            // Add a delete button for each entry
            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'Delete';
            // Basic styling - consider moving to CSS for better maintenance
            deleteBtn.style.cssText = `
                background-color: #ffc107;
                color: black;
                padding: 3px 8px;
                font-size: 12px;
                margin-left: 10px;
                float: right;
                border: none;
                border-radius: 3px;
                cursor: pointer;
            `;
            deleteBtn.onclick = (e) => {
                 e.stopPropagation(); // Prevent potential parent event triggers
                 deleteRegisteredFace(index);
            }
            li.appendChild(deleteBtn);

            registeredList.appendChild(li);
        });
    }
}

// --- 9. Delete Individual Face ---
function deleteRegisteredFace(indexToDelete) {
     if (indexToDelete < 0 || indexToDelete >= labeledFaceDescriptors.length) {
         console.error("Invalid index for deletion:", indexToDelete);
         return;
     }
     const deletedLabel = labeledFaceDescriptors[indexToDelete].label.split('|')[0];

     // Optional: Confirm before deleting individual entry
     if (confirm(`Are you sure you want to delete the registration for ${deletedLabel}?`)) {
         labeledFaceDescriptors.splice(indexToDelete, 1); // Remove from array
         console.log(`Deleted face at index ${indexToDelete} (${deletedLabel})`);

         // Update everything
         saveRegisteredFaces();
         updateFaceMatcher();
         updateRegisteredList();
         // Clear tracking as indices/matches might change
         trackedFaces = {};
     }
}


// --- 10. Clear All Registrations ---
clearAllButton.addEventListener('click', () => {
    if (confirm('Are you sure you want to delete ALL registered people? This cannot be undone.')) {
        labeledFaceDescriptors = []; // Clear array
        faceMatcher = null;        // Clear matcher
        localStorage.removeItem('faceRecognitionData'); // Clear storage
        updateRegisteredList();    // Update UI
        trackedFaces = {};         // Clear tracking
        clearStatus.textContent = 'All registrations cleared.';
        clearStatus.style.color = 'green';
         console.log("All registered faces cleared.");
        setTimeout(() => clearStatus.textContent = '', 3000); // Clear message after 3s
    }
});

// --- 11. Helper Functions ---

// Calculate distance between the centers of two boxes
function getCenterDistance(box1, box2) {
    if (!box1 || !box2) return Infinity; // Handle cases where boxes might be undefined
    const center1 = { x: box1.x + box1.width / 2, y: box1.y + box1.height / 2 };
    const center2 = { x: box2.x + box2.width / 2, y: box2.y + box2.height / 2 };
    const dx = center1.x - center2.x;
    const dy = center1.y - center2.y;
    return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Draws a rectangle with rounded corners.
 * @param {CanvasRenderingContext2D} ctx - The canvas rendering context.
 * @param {number} x - The top left x coordinate.
 * @param {number} y - The top left y coordinate.
 * @param {number} width - The width of the rectangle.
 * @param {number} height - The height of the rectangle.
 * @param {number} radius - The corner radius.
 */
function drawRoundedRect(ctx, x, y, width, height, radius) {
    // Ensure radius is not larger than half the shortest side
    radius = Math.min(radius, width / 2, height / 2);
    // Prevent negative radius
    radius = Math.max(0, radius);

    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
}


// --- Initial Load ---
loadModels(); // Start the application by loading models

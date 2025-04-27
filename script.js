// --- DOM Elements ---
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const loader = document.getElementById('loader');

// Mode Switching
const objectDetectionButton = document.getElementById('switchToObjectDetection');
const faceRecognitionButton = document.getElementById('switchToFaceRecognition');
const objectDetectionUI = document.getElementById('objectDetectionUI');
const faceRecognitionUI = document.getElementById('faceRecognitionUI');

// Object Detection Elements
const classInput = document.getElementById('classInput');
const updateClassesButton = document.getElementById('updateClassesButton');
const updateClassesStatus = document.getElementById('updateClassesStatus');
const detectingClassesList = document.getElementById('detectingClassesList');
const clearDetectedClassesButton = document.getElementById('clearDetectedClassesButton');
const clearClassesStatus = document.getElementById('clearClassesStatus');

// Face Recognition Elements
const nameInput = document.getElementById('nameInput');
const relationshipInput = document.getElementById('relationshipInput');
const imageUpload = document.getElementById('imageUpload');
const registerButton = document.getElementById('registerButton');
const registerStatus = document.getElementById('registerStatus');
const registeredList = document.getElementById('registeredList');
const clearAllButton = document.getElementById('clearAllButton');
const clearStatus = document.getElementById('clearStatus');


// --- Configuration ---
const COCO_SSD_MODEL_PATH = undefined; // Use default CDN path
const FACEAPI_MODEL_URL = './models'; // IMPORTANT: Assumes 'models' dir is in the same folder as index.html
const FACEAPI_DETECTOR_OPTIONS = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });
const DETECTION_INTERVAL = 150; // Interval (ms) for running detections in the loop
const FACE_MATCHER_THRESHOLD = 0.5; // Threshold for face matching (adjust as needed)

// --- Label Styling (Consolidated) ---
const TEXT_PADDING = 8; // Adjust padding as needed
const FONT_SIZE_OBJECT = 15;
const FONT_SIZE_FACE = 24;
// --- Updated Colors & Radius ---
const OBJECT_BG_COLOR = 'rgba(0, 0, 0, 0.3)';
const FACE_BG_COLOR = 'rgba(0, 0, 0, 0.3)';
// --- Font Color (White) ---
const FONT_COLOR = '#FFF';
// --- Font Style (Ferom, 36px, 400 weight, normal style) ---
const FONT_STYLE_OBJECT = `normal 400 ${FONT_SIZE_OBJECT}px Ferom, sans-serif`; // Fallback added
const FONT_STYLE_FACE = `normal 400 ${FONT_SIZE_FACE}px Ferom, sans-serif`;   // Fallback added
// --- NO Radius for Sharp Corners ---
const BACKGROUND_RADIUS = 30; // Commented out or removed
// --- Letter Spacing ---
const LETTER_SPACING = '-0.36px';


// --- State Variables ---
let currentMode = 'objectDetection'; // 'objectDetection' or 'faceRecognition'
let cocoSsdModel = null;
let faceMatcher = null;
let registeredFaceDescriptors = []; // Stores faceapi.LabeledFaceDescriptors
let allowedCocoClasses = []; // Array for COCO classes to detect
let modelsLoaded = { cocoSsd: false, faceApi: false };
let allModelsLoaded = false;
let videoPlaying = false;
let displaySize = null;
let detectionLoopInterval = null;
let lastDetectionTime = 0;
let currentFrameDetections = []; // Holds detections for the current frame/mode


// --- 0. Initialize TFJS Backend ---
async function initializeBackend() {
    try {
        if (typeof tf === 'undefined' || typeof tf.setBackend === 'undefined') {
            throw new Error("TensorFlow.js (tf) object or setBackend method not found!");
        }
        await tf.setBackend('webgl');
        await tf.ready();
        console.log(`TFJS backend set to: ${tf.getBackend()}`);
        return true;
    } catch (error) {
        console.error("Error setting TFJS backend:", error);
        loader.innerText = "Error initializing TFJS. See console.";
        loader.style.display = 'block';
        return false;
    }
}

// --- 1. Load All Models ---
async function loadAllModels() {
    console.log("Loading models...");
    loader.style.display = 'block';
    loader.innerText = "Initializing Backend...";

    const backendReady = await initializeBackend();
    if (!backendReady) return;

    loader.innerText = "Loading Models...";
    try {
        // Load COCO-SSD
        const cocoPromise = cocoSsd.load({ basePath: COCO_SSD_MODEL_PATH }).then(model => {
            cocoSsdModel = model;
            modelsLoaded.cocoSsd = true;
            console.log("COCO-SSD model loaded.");
            loadAllowedClasses(); // Load saved classes for object detection
        }).catch(err => { console.error("Failed to load COCO-SSD model:", err); throw err; });

        // Load FaceAPI models
        const faceApiPromise = Promise.all([
            faceapi.nets.ssdMobilenetv1.loadFromUri(FACEAPI_MODEL_URL),
            faceapi.nets.faceLandmark68Net.loadFromUri(FACEAPI_MODEL_URL),
            faceapi.nets.faceRecognitionNet.loadFromUri(FACEAPI_MODEL_URL)
        ]).then(() => {
            modelsLoaded.faceApi = true;
            console.log("FaceAPI models loaded.");
            loadRegisteredFaces(); // Load saved faces for recognition
        }).catch(err => { console.error("Failed to load FaceAPI models (check ./models path):", err); throw err;});

        // Wait for both
        await Promise.all([cocoPromise, faceApiPromise]);

        allModelsLoaded = true;
        loader.style.display = 'none';
        console.log("All models loaded successfully.");
        startVideo();

    } catch (error) {
        console.error("Error during model loading:", error);
        loader.innerText = `Error loading models. Check console.`;
        loader.style.display = 'block';
        allModelsLoaded = false;
    }
}

// --- 2. Start Webcam ---
function startVideo() {
    console.log("Attempting to start video...");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.error("getUserMedia not supported.");
        alert("Your browser does not support camera access.");
        return;
    }
    loader.innerText = "Requesting camera access...";
    loader.style.display = 'block';

    navigator.mediaDevices.getUserMedia({ video: {} })
        .then(stream => {
            console.log("Camera stream obtained.");
            video.srcObject = stream;
            video.addEventListener('loadedmetadata', () => {
                 console.log("Video metadata loaded.");
                 videoPlaying = true;
                 if (allModelsLoaded) {
                     startDetectionLoop();
                 } else {
                     console.warn("Metadata loaded, but models aren't ready?");
                 }
                 loader.style.display = 'none';
             });
            video.onerror = (e) => {
                 console.error("Video error:", e);
                 alert("Error playing video stream.");
                 loader.innerText = "Video Error.";
                 loader.style.display = 'block';
            };
        })
        .catch(err => {
            console.error("getUserMedia error:", err);
            alert(`Error accessing camera: ${err.name}. Check browser permissions.`);
            loader.innerText = `Camera Error: ${err.name}`;
            loader.style.display = 'block';
        });
}

// --- 3. Mode Switching Logic ---
function switchMode(newMode) {
    if (newMode === currentMode) return; // No change needed

    console.log(`Switching mode to: ${newMode}`);
    currentMode = newMode;
    currentFrameDetections = []; // Clear detections when switching mode
    const ctx = canvas.getContext('2d');
    if (ctx) ctx.clearRect(0, 0, canvas.width, canvas.height); // Clear canvas

    if (currentMode === 'objectDetection') {
        objectDetectionUI.style.display = 'flex'; // Or 'block' depending on layout needs
        faceRecognitionUI.style.display = 'none';
        objectDetectionButton.classList.add('active');
        faceRecognitionButton.classList.remove('active');
        loadAllowedClasses(); // Refresh UI list for current settings
    } else { // faceRecognition
        objectDetectionUI.style.display = 'none';
        faceRecognitionUI.style.display = 'flex'; // Or 'block'
        objectDetectionButton.classList.remove('active');
        faceRecognitionButton.classList.add('active');
        loadRegisteredFaces(); // Refresh UI list for current settings
    }

    // Restart or adjust the detection loop if necessary (optional, current loop handles modes)
    // if (detectionLoopInterval) {
    //     clearInterval(detectionLoopInterval);
    //     startDetectionLoop();
    // }
}

objectDetectionButton.addEventListener('click', () => switchMode('objectDetection'));
faceRecognitionButton.addEventListener('click', () => switchMode('faceRecognition'));


// --- 4. Start Main Detection Loop ---
function startDetectionLoop() {
    if (detectionLoopInterval) clearInterval(detectionLoopInterval);
    console.log("Starting combined detection loop.");

    const ctx = canvas.getContext('2d');

    // Setup canvas dimensions based on video
    const setupCanvas = () => {
        if (!videoPlaying || video.videoWidth === 0) return null;
        const currentDisplaySize = { width: video.clientWidth, height: video.clientHeight };

        if (!displaySize || displaySize.width !== currentDisplaySize.width || displaySize.height !== currentDisplaySize.height) {
            displaySize = currentDisplaySize;
            // Use faceapi utility for matching dimensions, works generally
            faceapi.matchDimensions(canvas, displaySize);
            console.log("Canvas dimensions synchronized:", displaySize);
            currentFrameDetections = []; // Clear boxes on resize
        }
        return displaySize;
    };
    displaySize = setupCanvas(); // Initial setup
    window.addEventListener('resize', () => { displaySize = null; setupCanvas(); }); // Re-setup on resize

    detectionLoopInterval = setInterval(async () => {
        if (video.paused || video.ended || !allModelsLoaded || !setupCanvas()) {
            return; // Stop if video not playing, models not loaded, or canvas not set up
        }

        const timeNow = Date.now();
        if (timeNow - lastDetectionTime < DETECTION_INTERVAL) {
            // Draw previous frame's detections if interval hasn't passed
            drawResults(ctx);
            return;
        }
        lastDetectionTime = timeNow;

        // --- Perform Detection Based on Mode ---
        if (currentMode === 'objectDetection' && modelsLoaded.cocoSsd && allowedCocoClasses.length > 0) {
            try {
                 const videoTensor = tf.browser.fromPixels(video);
                 const objectPredictions = await cocoSsdModel.detect(videoTensor);
                 videoTensor.dispose(); // IMPORTANT: Dispose tensor
                 currentFrameDetections = objectPredictions.filter(p =>
                    allowedCocoClasses.includes(p.class.toLowerCase()) // Ensure lowercase comparison
                 );
            } catch (objDetError) {
                console.error("Error during COCO-SSD detection:", objDetError);
                currentFrameDetections = [];
            }
        } else if (currentMode === 'faceRecognition' && modelsLoaded.faceApi && faceMatcher) {
            try {
                const detections = await faceapi.detectAllFaces(video, FACEAPI_DETECTOR_OPTIONS)
                                              .withFaceLandmarks()
                                              .withFaceDescriptors();

                if (detections.length > 0 && displaySize) {
                     const resizedDetections = faceapi.resizeResults(detections, displaySize);
                     currentFrameDetections = resizedDetections.map(d => {
                         const bestMatch = faceMatcher.findBestMatch(d.descriptor);
                         return { detection: d.detection, match: bestMatch };
                     });
                } else {
                     currentFrameDetections = [];
                }
            } catch (faceError) {
                console.error("Error during FaceAPI detection:", faceError);
                currentFrameDetections = [];
            }
        } else {
            // No detections needed for this mode or prerequisites not met
            currentFrameDetections = [];
        }

        // --- Drawing ---
        drawResults(ctx);

    }, 50); // Draw loop runs faster than detection interval

    // Cleanup on pause/end
    const stopLoop = () => {
         if (detectionLoopInterval) clearInterval(detectionLoopInterval);
         videoPlaying = false;
         console.log("Detection loop stopped.");
         const clearCtx = canvas.getContext('2d');
         if(clearCtx) clearCtx.clearRect(0, 0, canvas.width, canvas.height);
         currentFrameDetections = [];
    };
    video.addEventListener('pause', stopLoop);
    video.addEventListener('ended', stopLoop);
}

// --- 5. Drawing Results ---
// --- 5. Drawing Results ---
function drawResults(ctx) {
    if (!displaySize) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // --- Pre-calculate approx text height based on font size ---
    // Note: This duplicates logic from drawLabel but is needed here for positioning calculations.
    // We assume object/face use same font size for simplicity here, adjust if needed.
    const tempFontSize = FONT_SIZE_OBJECT; // Use one of the font sizes
    const approxTextHeight = tempFontSize * 1.2; // Sync with drawLabel's calculation
    const tempPadding = TEXT_PADDING; // Sync with drawLabel

    if (currentMode === 'objectDetection') {
        currentFrameDetections.forEach(detection => {
            const [x, y, width, height] = detection.bbox;
            const label = `${detection.class} (${detection.score.toFixed(2)})`;

            // --- Calculate Label Dimensions (duplicate from drawLabel) ---
            ctx.font = FONT_STYLE_OBJECT; // Need to set font to measure accurately
            const metrics = ctx.measureText(label);
            const textWidth = metrics.width;
            const rectWidth = textWidth + (2 * tempPadding);
            const rectHeight = approxTextHeight + (2 * tempPadding);
            // --- End Dimension Calculation ---

            // --- Position Label Centered INSIDE the box ---
            const labelX = x + (width / 2) - (rectWidth / 2);
            const labelY = y + (height / 2) - (rectHeight / 2);
            // --- End Positioning ---

            const labelPos = { x: labelX, y: labelY };
            drawLabel(ctx, labelPos, label, OBJECT_BG_COLOR, FONT_STYLE_OBJECT, tempFontSize); // Pass size used for calculation
            // Bounding box drawing is still commented out
        });
    } else if (currentMode === 'faceRecognition') {
        currentFrameDetections.forEach(result => {
            const box = result.detection.box;
            const bestMatch = result.match;

            // Parse the label (keep logic from previous step)
            let displayText = bestMatch.toString();
            if (bestMatch.label !== 'unknown') {
                const parts = bestMatch.label.split(':::');
                const name = parts[0];
                const relationship = parts.length > 1 ? parts[1] : null;
                displayText = name;
                if (relationship) {
                    displayText += ` | ${relationship}`;
                }
                // Optional: Add distance
                // displayText += ` (${bestMatch.distance.toFixed(2)})`;
            }

            // --- Calculate Label Dimensions (duplicate from drawLabel) ---
             ctx.font = FONT_STYLE_FACE; // Need to set font to measure accurately
             const metrics = ctx.measureText(displayText);
             const textWidth = metrics.width;
             const rectWidth = textWidth + (2 * tempPadding);
             const rectHeight = approxTextHeight + (2 * tempPadding);
            // --- End Dimension Calculation ---

            // --- Position Label Centered ABOVE the box ---
            const labelX = box.x + (box.width / 2) - (rectWidth / 2); // Center horizontally
            const labelY = box.y - rectHeight - 4; // Position above box top edge with a 4px gap
            // --- End Positioning ---

            const labelPos = { x: labelX, y: labelY };
            drawLabel(ctx, labelPos, displayText, FACE_BG_COLOR, FONT_STYLE_FACE, tempFontSize); // Pass size used for calculation
             // Bounding box drawing is still commented out
        });
    }
    // Reset font potentially changed during measurement
    ctx.font = '10px sans-serif'; // Or whatever default you prefer
}

// Generic Label Drawing Function
function drawLabel(ctx, pos, text, bgColor, fontStyle, fontSize) {
    ctx.font = fontStyle;
    ctx.letterSpacing = LETTER_SPACING;

    const metrics = ctx.measureText(text);
    const textWidth = metrics.width;
    const textHeight = fontSize * 1.4; // Approx height based on 140% line-height
    const rectWidth = textWidth + (2 * TEXT_PADDING);
    const rectHeight = textHeight + (2 * TEXT_PADDING);

    let rectX = pos.x;
    let rectY = pos.y;

    const finalRectX = Math.max(0, Math.min(canvas.width - rectWidth, rectX));
    const finalRectY = Math.max(0, Math.min(canvas.height - rectHeight, rectY));

    ctx.fillStyle = bgColor;
    // --- Use drawRoundedRect and fill() ---
    drawRoundedRect(ctx, finalRectX, finalRectY, rectWidth, rectHeight, BACKGROUND_RADIUS);
    ctx.fill(); // Fill the rounded rectangle path

    ctx.fillStyle = FONT_COLOR;
    ctx.textBaseline = 'middle';
    const textX = finalRectX + TEXT_PADDING;
    const textY = finalRectY + rectHeight / 2;
    ctx.fillText(text, textX, textY);

    ctx.letterSpacing = '0px'; // Reset
    ctx.textBaseline = 'alphabetic'; // Reset
}

// --- 6. Object Detection UI Logic ---
updateClassesButton.addEventListener('click', () => {
    updateClassesStatus.textContent = '';
    const inputText = classInput.value.trim();
    if (!inputText) {
        allowedCocoClasses = []; // Clear detection list
        updateClassesStatus.textContent = 'Detection list cleared.';
        updateClassesStatus.style.color = 'orange';
    } else {
        allowedCocoClasses = inputText
            .split(',')
            .map(cls => cls.trim().toLowerCase())
            .filter(cls => cls); // Remove empty strings
        updateClassesStatus.textContent = `Detecting: ${allowedCocoClasses.join(', ')}`;
         updateClassesStatus.style.color = 'green';
    }
    saveAllowedClasses();
    updateDetectingClassesListUI();
    currentFrameDetections = []; // Clear current boxes
    setTimeout(() => updateClassesStatus.textContent = '', 4000);
});

clearDetectedClassesButton.addEventListener('click', () => {
    allowedCocoClasses = [];
    classInput.value = '';
    saveAllowedClasses();
    updateDetectingClassesListUI();
    currentFrameDetections = [];
    clearClassesStatus.textContent = 'Selection cleared.';
    clearClassesStatus.style.color = 'green';
    setTimeout(() => clearClassesStatus.textContent = '', 3000);
});

function updateDetectingClassesListUI() {
    detectingClassesList.innerHTML = '';
    if (allowedCocoClasses.length === 0) {
        detectingClassesList.innerHTML = '<li>No classes selected.</li>';
    } else {
        allowedCocoClasses.forEach((cls) => {
            const li = document.createElement('li');
            li.textContent = cls;
            detectingClassesList.appendChild(li);
        });
    }
}

// --- 7. Face Recognition UI Logic ---
registerButton.addEventListener('click', async () => {
    registerStatus.textContent = ''; registerStatus.style.color = 'black';
    if (!modelsLoaded.faceApi) {
        registerStatus.textContent = 'Face models not ready.'; registerStatus.style.color = 'red'; return;
    }
    const name = nameInput.value.trim(); // Use 'name' variable
    // --- Get relationship value ---
    const relationship = relationshipInput.value.trim();
    const imageFile = imageUpload.files[0];

    // --- Check if name is provided ---
    if (!name || !imageFile) {
        registerStatus.textContent = 'Please provide a name and image file.'; registerStatus.style.color = 'red'; return;
    }

    // --- Construct the combined label ---
    let combinedLabel = name;
    if (relationship) {
        combinedLabel += `:::${relationship}`; // Use separator
    }
    // --- End combined label construction ---


    // Check for duplicates (using the NAME part only for confirmation message)
    // Note: We still store the combinedLabel, but check based on name for user confirmation.
    const existingIndex = registeredFaceDescriptors.findIndex(lfd => lfd.label.startsWith(name + ':::') || lfd.label === name);
    if (existingIndex !== -1) {
        // Use the extracted name for the confirmation message
         if (!confirm(`"${name}" is already registered. Overwrite?`)) {
            registerStatus.textContent = 'Registration cancelled.'; registerStatus.style.color = 'orange'; return;
         }
         registeredFaceDescriptors.splice(existingIndex, 1);
         console.log(`Overwriting registration for: ${name}`);
    }


    registerStatus.textContent = 'Processing image...'; registerStatus.style.color = 'orange';
    try {
        const imageElement = await faceapi.bufferToImage(imageFile);
        registerStatus.textContent = 'Detecting face & creating descriptor...';
        const detection = await faceapi.detectSingleFace(imageElement, FACEAPI_DETECTOR_OPTIONS)
                                     .withFaceLandmarks().withFaceDescriptor();

        if (!detection || !detection.descriptor) {
             registerStatus.textContent = detection ? 'Could not compute descriptor.' : 'No face detected in image.';
             registerStatus.style.color = 'red'; return;
        }

        // Create and add the new descriptor using the COMBINED label
        // --- Use combinedLabel here ---
        const newDescriptor = new faceapi.LabeledFaceDescriptors(combinedLabel, [detection.descriptor]);
        registeredFaceDescriptors.push(newDescriptor);
        console.log(`Registered "${combinedLabel}"`); // Log combined label

        updateFaceMatcher();
        saveRegisteredFaces();
        updateRegisteredListUI(); // Update the displayed list (might need adjustment, see step 3)

        // --- Use name for status message ---
        registerStatus.textContent = `Registered "${name}" successfully!`;
        registerStatus.style.color = 'green';
        nameInput.value = '';
        relationshipInput.value = ''; // Clear relationship input too
        imageUpload.value = '';
    } catch (error) {
        console.error("Registration error:", error);
        registerStatus.textContent = `Registration failed: ${error.message}`;
        registerStatus.style.color = 'red';
    } finally {
        setTimeout(() => { if (registerStatus.style.color !== 'red') registerStatus.textContent = ''; }, 5000);
    }
});

clearAllButton.addEventListener('click', () => {
    if (registeredFaceDescriptors.length > 0 && confirm('Delete ALL registered faces? This cannot be undone.')) {
        registeredFaceDescriptors = [];
        faceMatcher = null; // Clear the matcher
        localStorage.removeItem('faceRecognitionData_faceapi_v2'); // Clear storage
        updateRegisteredListUI(); // Update UI
        clearStatus.textContent = 'All registrations cleared.';
        clearStatus.style.color = 'green';
        console.log("All registered faces cleared.");
        setTimeout(() => clearStatus.textContent = '', 3000);
    } else if (registeredFaceDescriptors.length === 0) {
        clearStatus.textContent = 'No faces registered to clear.';
        clearStatus.style.color = 'orange';
        setTimeout(() => clearStatus.textContent = '', 3000);
    }
});


function updateRegisteredListUI() {
    registeredList.innerHTML = '';
    if (registeredFaceDescriptors.length === 0) {
        registeredList.innerHTML = '<li>No faces registered yet.</li>';
    } else {
        registeredFaceDescriptors.forEach((lfd, index) => {
            const li = document.createElement('li');
            // --- Extract name for display ---
            const labelParts = lfd.label.split(':::');
            const displayName = labelParts[0]; // Always show the name part
            // --- End name extraction ---
            li.textContent = displayName; // Display only name in the list

            const deleteBtn = document.createElement('button');
            deleteBtn.textContent = 'Delete';
            deleteBtn.onclick = (e) => {
                e.stopPropagation();
                deleteSingleRegisteredFace(index); // Deletion logic might need check
            };
            li.appendChild(deleteBtn);
            registeredList.appendChild(li);
        });
    }
    updateFaceMatcher();
}

function deleteSingleRegisteredFace(indexToDelete) {
     if (indexToDelete < 0 || indexToDelete >= registeredFaceDescriptors.length) return;
     // --- Extract name for confirmation ---
     const labelToDelete = registeredFaceDescriptors[indexToDelete].label;
     const nameToDelete = labelToDelete.split(':::')[0];
     // --- Use extracted name ---
     if (confirm(`Delete registration for "${nameToDelete}"?`)) {
         const deletedLabel = registeredFaceDescriptors[indexToDelete].label; // Store full label before splice
         registeredFaceDescriptors.splice(indexToDelete, 1);
         saveRegisteredFaces();
         updateRegisteredListUI();
         console.log(`Deleted registration: ${deletedLabel}`); // Log full deleted label
         registerStatus.textContent = `Deleted "${nameToDelete}".`;
         registerStatus.style.color = 'orange';
         setTimeout(() => registerStatus.textContent = '', 3000);
     }
}


function updateFaceMatcher() {
    if (!modelsLoaded.faceApi || registeredFaceDescriptors.length === 0) {
        faceMatcher = null;
        console.log("Face matcher cleared or not ready.");
        return;
    }
    try {
         faceMatcher = new faceapi.FaceMatcher(registeredFaceDescriptors, FACE_MATCHER_THRESHOLD);
         console.log("Face matcher updated with", registeredFaceDescriptors.length, "descriptors.");
    } catch (error) {
        console.error("Error creating FaceMatcher:", error);
        faceMatcher = null;
    }
}


// --- 8. Persistence ---
// Use distinct keys to avoid conflicts between modes
const COCO_CLASSES_STORAGE_KEY = 'combinedApp_cocoClasses';
const FACE_DESCRIPTORS_STORAGE_KEY = 'combinedApp_faceDescriptors';

function saveAllowedClasses() {
    try {
        localStorage.setItem(COCO_CLASSES_STORAGE_KEY, JSON.stringify(allowedCocoClasses));
    } catch (error) { console.error("Error saving allowed classes:", error); }
}

function loadAllowedClasses() {
    const savedData = localStorage.getItem(COCO_CLASSES_STORAGE_KEY);
    if (savedData) {
        try {
            const parsedData = JSON.parse(savedData);
            allowedCocoClasses = Array.isArray(parsedData) ? parsedData : [];
        } catch (error) { console.error("Failed to parse saved classes:", error); allowedCocoClasses = []; }
    } else {
        allowedCocoClasses = []; // Default to empty if nothing saved
    }
    classInput.value = allowedCocoClasses.join(', '); // Update input field
    updateDetectingClassesListUI(); // Update UI list
}

function saveRegisteredFaces() {
     if (typeof faceapi === 'undefined') return; // Need faceapi to serialize
     try {
         // Convert LabeledFaceDescriptors to a serializable format
         const serializableData = registeredFaceDescriptors.map(lfd => lfd.toJSON());
         localStorage.setItem(FACE_DESCRIPTORS_STORAGE_KEY, JSON.stringify(serializableData));
         console.log("Registered faces saved.");
     } catch (error) { console.error("Error saving face data:", error); }
}

function loadRegisteredFaces() {
    if (typeof faceapi === 'undefined') return; // Need faceapi to deserialize
    const savedData = localStorage.getItem(FACE_DESCRIPTORS_STORAGE_KEY);
    registeredFaceDescriptors = []; // Start fresh
    if (savedData) {
        try {
            const parsedData = JSON.parse(savedData);
            if (Array.isArray(parsedData)) {
                // Convert back from serialized format to LabeledFaceDescriptors
                registeredFaceDescriptors = parsedData.map(item => faceapi.LabeledFaceDescriptors.fromJSON(item));
                console.log("Loaded", registeredFaceDescriptors.length, "registered faces.");
            }
        } catch (error) {
             console.error("Failed to parse saved face data:", error);
             localStorage.removeItem(FACE_DESCRIPTORS_STORAGE_KEY); // Clear corrupted data
        }
    }
    updateRegisteredListUI(); // Update the UI list
    updateFaceMatcher();    // Update the matcher with loaded faces
}

// --- 9. Helper Function ---
function drawRoundedRect(ctx, x, y, width, height, radius) {
    if (width <= 0 || height <= 0) return; // Avoid drawing invalid rects
    radius = Math.min(radius, width / 2, height / 2);
    radius = Math.max(0, radius); // Ensure radius is not negative
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.arcTo(x + width, y, x + width, y + height, radius);
    ctx.arcTo(x + width, y + height, x, y + height, radius);
    ctx.arcTo(x, y + height, x, y, radius);
    ctx.arcTo(x, y, x + width, y, radius);
    ctx.closePath();
}

// --- Initial Load ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded. Checking libraries...");
    if (typeof tf !== 'undefined' && typeof cocoSsd !== 'undefined' && typeof faceapi !== 'undefined') {
        console.log("Libraries seem available. Starting model loading...");
        loadAllModels();
    } else {
        console.error("One or more libraries (tf, cocoSsd, faceapi) failed to load!");
        loader.innerText = "Error: Required libraries missing. Check console.";
        loader.style.display = 'block';
        alert("Critical error: Could not load necessary AI libraries. Check your internet connection and browser console.");
    }
});
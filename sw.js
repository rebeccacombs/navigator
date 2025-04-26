const CACHE_NAME = 'navigator-recog-cache-v1';

// --- Files to Cache ---

// App Shell: Core files needed to run the app UI
const APP_SHELL_FILES = [
  './', // Cache the root path if it serves index.html
  './index.html',
  './style.css',
  './script.js',
  './face-api.min.js',
  './manifest.json', // Cache the manifest itself
  // Add paths to your icons here:
  './icons/icon-192x192.png',
  './icons/icon-512x512.png'
];

// Models: Essential face-api.js models (adjust paths if needed)
// Note: Caching many large model files can take up significant space.
// Consider caching only the models you actually load in script.js.
const MODEL_FILES = [
  // --- SSD Mobilenet V1 (Detection) ---
  './models/ssd_mobilenetv1_model-weights_manifest.json',
  './models/ssd_mobilenetv1_model-shard1',
  './models/ssd_mobilenetv1_model-shard2',

  // --- Face Landmark 68 Net (Landmarks) ---
  './models/face_landmark_68_model-weights_manifest.json',
  './models/face_landmark_68_model-shard1',

  // --- Face Recognition Net (Descriptors) ---
  './models/face_recognition_model-weights_manifest.json',
  './models/face_recognition_model-shard1',
  './models/face_recognition_model-shard2',

  // Add other models if you use them (Tiny models, Age/Gender, etc.)
  // e.g., './models/tiny_face_detector_model-weights_manifest.json',
  //       './models/tiny_face_detector_model-shard1', ...
];

const ALL_FILES_TO_CACHE = [...APP_SHELL_FILES, ...MODEL_FILES];

// --- Service Worker Lifecycle ---

// Install: Cache essential files when the SW is first installed
self.addEventListener('install', (event) => {
  console.log('[Service Worker] Install event');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[Service Worker] Caching app shell and models');
        // Use addAll for atomic caching - if one file fails, the whole operation fails.
        return cache.addAll(ALL_FILES_TO_CACHE);
      })
      .then(() => {
        console.log('[Service Worker] Caching complete. Ready for offline use.');
        // Force the waiting service worker to become the active service worker.
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('[Service Worker] Caching failed:', error);
      })
  );
});

// Activate: Clean up old caches
self.addEventListener('activate', (event) => {
  console.log('[Service Worker] Activate event');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('[Service Worker] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('[Service Worker] Claiming clients');
      // Ensure the activated SW takes control immediately
      return self.clients.claim();
    })
  );
});

// Fetch: Intercept network requests and serve from cache if available
self.addEventListener('fetch', (event) => {
  // console.log('[Service Worker] Fetching:', event.request.url);

  // Use a Cache-First strategy for all cached assets
  event.respondWith(
    caches.match(event.request) // Check if the request matches a cached asset
      .then((cachedResponse) => {
        if (cachedResponse) {
          // console.log('[Service Worker] Serving from cache:', event.request.url);
          return cachedResponse; // Serve the cached version
        }
        // console.log('[Service Worker] Fetching from network:', event.request.url);
        return fetch(event.request); // If not in cache, fetch from the network
        // Optional: Cache newly fetched resources dynamically (use with caution for large files)
        /*
        .then(networkResponse => {
            // Check if we received a valid response
            if(!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
              return networkResponse;
            }
            // IMPORTANT: Cloning the response. A response is a stream
            // and because we want the browser to consume the response
            // as well as the cache consuming the response, we need
            // to clone it so we have two streams.
            const responseToCache = networkResponse.clone();

            caches.open(CACHE_NAME)
              .then(cache => {
                console.log('[Service Worker] Caching new resource:', event.request.url);
                cache.put(event.request, responseToCache);
              });

            return networkResponse;
        })
        */
      })
      .catch(error => {
        console.error('[Service Worker] Fetch failed:', error);
        // Optional: Provide a fallback page for offline navigation errors
        // if (event.request.mode === 'navigate') {
        //   return caches.match('./offline.html'); // You would need to create and cache offline.html
        // }
      })
  );
});
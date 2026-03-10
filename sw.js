// W@Home Service Worker — Cache Pyodide + packages for instant subsequent loads
const CACHE_NAME = 'whome-v2';

// Core app files to precache
const PRECACHE = [
  '/',
  '/compute',
  '/manifest.json',
];

// Patterns for immutable resources (cache-first: serve from cache, never re-download)
const IMMUTABLE_PATTERNS = [
  '/pyodide/',                              // locally-hosted Pyodide + numpy + scipy
  'cdnjs.cloudflare.com/ajax/libs/three.js', // Three.js CDN (background animation)
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(PRECACHE))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys
        .filter(k => k !== CACHE_NAME)
        .map(k => caches.delete(k))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  const url = event.request.url;

  // Don't cache API calls (POST requests, dynamic endpoints)
  if (event.request.method !== 'GET') return;
  if (url.includes('/job') || url.includes('/result') || url.includes('/register') ||
      url.includes('/heartbeat') || url.includes('/progress') || url.includes('/workers') ||
      url.includes('/leaderboard') || url.includes('/active') || url.includes('/worker/update') ||
      url.includes('/dashboard') || url.includes('/discoveries')) {
    return;
  }

  // Cache-first for Pyodide/Three.js (immutable, versioned — never changes)
  const isImmutable = IMMUTABLE_PATTERNS.some(p => url.includes(p));
  if (isImmutable) {
    event.respondWith(
      caches.match(event.request).then(cached => {
        if (cached) return cached;
        return fetch(event.request).then(response => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
          }
          return response;
        });
      })
    );
    return;
  }

  // Network-first for app files (compute.html, etc. — so updates propagate)
  event.respondWith(
    fetch(event.request)
      .then(response => {
        if (response.ok) {
          const clone = response.clone();
          caches.open(CACHE_NAME).then(cache => cache.put(event.request, clone));
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});

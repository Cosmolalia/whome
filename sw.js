// W@Home Service Worker — Cache Pyodide + packages for instant subsequent loads
const CACHE_NAME = 'whome-v1';

// Core app files to precache
const PRECACHE = [
  '/',
  '/compute',
  '/manifest.json',
];

// CDN resources to cache on first use (Pyodide + Three.js)
const CDN_PATTERNS = [
  'cdn.jsdelivr.net/pyodide',
  'cdnjs.cloudflare.com/ajax/libs/three.js',
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

  // Don't cache API calls (POST requests, /job, /result, /register, etc.)
  if (event.request.method !== 'GET') return;
  if (url.includes('/job') || url.includes('/result') || url.includes('/register') ||
      url.includes('/heartbeat') || url.includes('/progress') || url.includes('/workers') ||
      url.includes('/leaderboard') || url.includes('/active') || url.includes('/worker/update')) {
    return;
  }

  // Cache-first for CDN resources (Pyodide, Three.js — these are versioned/immutable)
  const isCDN = CDN_PATTERNS.some(p => url.includes(p));
  if (isCDN) {
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

  // Network-first for app files (so updates propagate)
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

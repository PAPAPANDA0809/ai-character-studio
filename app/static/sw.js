// Reelmint's service worker: only shows the "job finished" notifications (push.py) and opens the project when one
// is tapped. It doesn't cache or intercept anything.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', e => e.waitUntil(self.clients.claim()));

self.addEventListener('push', e => {
  let d = {};
  try { d = e.data ? e.data.json() : {}; } catch (err) { d = {body: e.data && e.data.text()}; }
  e.waitUntil(self.registration.showNotification(d.title || 'Reelmint', {
    body: d.body || '', icon: '/static/reelmint-192.png', tag: d.tag, renotify: true, data: {url: d.url || '/'}}));
});

self.addEventListener('notificationclick', e => {
  e.notification.close();
  const url = new URL(e.notification.data.url || '/', self.location.origin).href;
  e.waitUntil(self.clients.matchAll({type: 'window', includeUncontrolled: true}).then(list => {
    const open = list.find(c => c.url.startsWith(self.location.origin));
    if (open) return open.focus().then(c => c.navigate(url));
    return self.clients.openWindow(url);
  }));
});

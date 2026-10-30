const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../app/static/index.html'), 'utf8');
const source = html.slice(html.indexOf('const picItem ='), html.indexOf('async function openPicker('));
const calls = [];
const ctx = vm.createContext({
  CHARS: ['my-face.png'], PICLIB: {face: {items: [{id:'face:face1', cat:'Face portraits', sub:'Short'}]}},
  api: async (url, body) => { calls.push({url, body}); return body ? {name:'gallery-face1.png'} : ['my-face.png', 'gallery-face1.png']; },
});
vm.runInContext(source, ctx);
assert.equal(vm.runInContext('faceLib().items.length', ctx), 2);
assert.equal(vm.runInContext('picThumb("face:face1", "face")', ctx), '/library/thumb/face%3Aface1');
assert.equal(vm.runInContext('picThumb("my-face.png", "face")', ctx), '/characters/my-face.png');
assert.equal(vm.runInContext('picThumb("pose:P1", "pose")', ctx), '/library/thumb/pose%3AP1');
(async () => {
  assert.equal(await ctx.resolveFacePicture('my-face.png'), 'my-face.png');
  assert.equal(calls.length, 0, 'Existing references require no gallery import');
  assert.equal(await ctx.resolveFacePicture('face:face1'), 'gallery-face1.png');
  assert.equal(calls[0].url, '/api/library/face/select');
  assert.equal(calls[0].body.id, 'face:face1');
  assert.equal(calls[1].url, '/api/characters');
  assert.ok(ctx.CHARS.includes('gallery-face1.png'), 'Refresh private characters before applying the reference');
  console.log('Face gallery: combined picker, thumbnail routing and reference selection passed.');
})().catch(error => {console.error(error);process.exitCode=1;});

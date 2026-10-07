// Exercise the shared gesture handler without a browser or user media.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const html = fs.readFileSync(path.join(__dirname, '../app/static/index.html'), 'utf8');
const source = html.slice(html.indexOf('function makeZoom(img)'), html.indexOf('function lightbox('));

function viewer() {
  const listeners = new Map(), classes = new Set(), swipes = [];
  const box = {
    clientWidth:200, clientHeight:320,
    appendChild() {}, setPointerCapture() {},
    getBoundingClientRect:() => ({left:0, top:0, width:200, height:320}),
    classList:{toggle:(name, on) => on ? classes.add(name) : classes.delete(name)},
    addEventListener(name, callback) { listeners.set(name, callback); },
    dispatchEvent(event) { if (event.type === 'swipe') swipes.push(event.detail); listeners.get(event.type)?.(event); }
  };
  const context = vm.createContext({document:{createElement:() => box},
    CustomEvent:class { constructor(type, options) { this.type=type; this.detail=options.detail; } }});
  vm.runInContext(source, context);
  context.makeZoom({style:{}, replaceWith() {}});
  const pointer = (type, x, y=160, id=1, pointerType='touch') => box['on' + type]({type, clientX:x, clientY:y, pointerId:id, pointerType, button:0});
  const drag = (from, to, pointerType='touch') => {
    pointer('pointerdown', from, 160, 1, pointerType);
    pointer('pointermove', to, 160, 1, pointerType);
    pointer('pointerup', to, 160, 1, pointerType);
  };
  const click = () => {
    let blocked = false;
    box.dispatchEvent({type:'click', preventDefault() { blocked=true; }, stopImmediatePropagation() {}});
    return blocked;
  };
  return {box, swipes, pointer, drag, click};
}

for (const pointerType of ['touch', 'mouse', 'pen']) {
  const v = viewer(); v.drag(160, 60, pointerType); v.drag(60, 160, pointerType);
  assert.deepEqual(v.swipes, [1,-1], `${pointerType}: left and right navigation`);
}
{
  const v=viewer(); v.drag(120, 100);
  v.pointer('pointerdown',100,100); v.pointer('pointermove',90,250); v.pointer('pointerup',90,250);
  assert.deepEqual(v.swipes, [], 'short and vertical gestures do not navigate');
}
{
  const v=viewer(); v.drag(160,60);
  assert.equal(v.click(),true,'swiping does not trigger a tap/full screen');
  v.pointer('pointerdown',100); v.pointer('pointerup',100);
  assert.equal(v.click(),false,'a subsequent tap still works');
}
{
  const v=viewer(); v.box.zoomTo(2); v.drag(150,50);
  assert.deepEqual(v.swipes, [], 'a zoomed drag pans until it reaches the edge');
  assert.equal(v.box.edges().right,true);
  v.drag(150,50);
  assert.deepEqual(v.swipes,[1],'swiping from the right edge goes to the next picture');
  v.drag(20,250); // pan back to the left edge
  assert.deepEqual(v.swipes,[1],'reaching an edge does not also change pictures');
  v.drag(20,120);
  assert.deepEqual(v.swipes,[1,-1],'swiping from the left edge goes to the previous picture');
}
{
  const v=viewer();
  v.pointer('pointerdown',50,160,1); v.pointer('pointerdown',150,160,2);
  v.pointer('pointermove',250,160,2);
  assert.equal(v.box.scale(),2,'pinching still zooms');
  v.pointer('pointerup',250,160,2); v.pointer('pointermove',150,160,1); v.pointer('pointerup',150,160,1);
  assert.deepEqual(v.swipes,[],'a pinch or its remaining finger never changes pictures');
  assert.equal(v.click(),true,'a pinch does not also activate full screen');
}
{
  const v=viewer(); v.pointer('pointerdown',160); v.pointer('pointermove',60); v.pointer('pointercancel',60);
  assert.deepEqual(v.swipes,[],'cancelled gestures do not navigate');
  v.drag(160,60); assert.deepEqual(v.swipes,[1],'navigation recovers after cancellation');
}
console.log('Image gestures: touch/mouse/pen, both directions, pan edges, pinch, cancellation and tap isolation passed');

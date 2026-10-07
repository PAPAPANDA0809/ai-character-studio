// Verify actual gesture helpers, including suppression of destructive/costly actions after holding an icon.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const html = fs.readFileSync(path.join(__dirname,'../app/static/index.html'),'utf8');
const source = html.slice(html.indexOf('function bindTrayDrag('),html.indexOf('function bindActionHints('));
function fixture(open=false) {
  const listeners = new Map(), timers = new Map(); let shown=0, hidden=0, next=0;
  const target = {offsetHeight:52,setPointerCapture(){},addEventListener(n,fn){listeners.set(n,fn)}};
  const drawer = {offsetHeight:350,style:{}};
  const ctx = vm.createContext({STCONTROLS:open,getComputedStyle:()=>({paddingBottom:'0'}),
    setStudioControls(value){ctx.STCONTROLS=value},setTimeout(fn){timers.set(++next,fn);return next},clearTimeout(id){timers.delete(id)}});
  vm.runInContext(source,ctx);
  const fire = (type,options={}) => {
    const e = {pointerType:'touch',pointerId:1,clientX:100,clientY:650,isPrimary:true,button:0,...options,
      prevented:false,stopped:false,preventDefault(){this.prevented=true},stopImmediatePropagation(){this.stopped=true}};
    listeners.get(type)?.(e);return e;
  };
  return {ctx, target, drawer,fire,timers,hold(){ctx.bindHoldLabel(target,()=>shown++,()=>hidden++)},
    drag(){ctx.bindTrayDrag(target,drawer)},flush(){for(const fn of timers.values())fn();timers.clear()},shown:()=>shown};
}
{
  const f=fixture();f.hold();f.fire('pointerdown');f.flush();f.fire('pointerup');
  assert.equal(f.shown(),1);assert.equal(f.fire('click').stopped,true,'Holding must not trigger the action');
  f.fire('pointerdown');f.fire('pointerup');assert.equal(f.fire('click').stopped,false,'A normal tap still works');
}
{
  const f=fixture();f.hold();f.fire('pointerdown');f.fire('pointermove',{clientX:130});f.flush();
  assert.equal(f.shown(),0,'Scrolling cancels the hold');f.fire('pointercancel');assert.equal(f.fire('click').stopped,false);
}
{
  const f=fixture();f.drag();f.fire('pointerdown');f.fire('pointermove',{clientY:520});f.fire('pointerup',{clientY:520});
  assert.equal(f.ctx.STCONTROLS,true);assert.equal(f.fire('click').stopped,true,'Drag release must not toggle again');
}
{
  const f=fixture(true);f.drag();f.fire('pointerdown');f.fire('pointermove',{clientY:730});f.fire('pointerup',{clientY:730});
  assert.equal(f.ctx.STCONTROLS,false);
}
{
  const f=fixture();f.drag();f.fire('pointerdown');f.fire('pointermove',{clientY:520});f.fire('pointercancel');
  assert.equal(f.ctx.STCONTROLS,false,'Cancelled drag returns to original state');
  assert.equal(f.drawer.style.transform,'');
}
console.log('Studio controls: hold safety, normal taps, scrolling, drag open/close and cancellation passed.');

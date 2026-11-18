"""Weighted-only displays; all registered category-removal/reciprocal effects.
Conclusion: category removal and reciprocal composition have task- and learner-specific effects.
Data: frozen paired differences; n=3 learner/data blocks; center=mean, bars=sample SD.
No inferential labels. Primary native final success, common 1,024-reset bank per task.
Archetype: three aligned task panels. Exports: 180 mm, editable PDF/SVG and 300 dpi PNG.
"""
from pathlib import Path
import json,sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
ROOT=Path(__file__).parent;OUT=ROOT/'figures';OUT.mkdir(exist_ok=True)
rows=json.loads((ROOT/'weighted_contrasts_audited.json').read_text())
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.dpi':300,'axes.spines.top':False,'axes.spines.right':False})
tasks=['FetchPush','PointMaze','AntMazeLarge'];names=['FetchPush','PointMaze','AntMaze'];used=[];qa={}
def one(task,kind,method,axis=None):
 rr=[r for r in rows if r['task']==task and kind in r['family'] and r['method']==method and (axis is None or r.get('axis')==axis)]
 assert len(rr)==1,(task,kind,method,axis,len(rr));used.append(rr[0]);return rr[0]
def save(fig,name):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
 for t in fig.findobj(matplotlib.text.Text):
  if not t.get_visible() or not t.get_text():continue
  b=t.get_window_extent(renderer)
  if b.width and b.height and (b.x0<0 or b.y0<0 or b.x1>fig.bbox.width or b.y1>fig.bbox.height):outside.append(t.get_text())
 for ax in fig.axes:
  for c in ax.containers:
   if isinstance(c,matplotlib.container.ErrorbarContainer):
    for bar in c.lines[2]:
     for segment in bar.get_segments():
      assert all(min(ax.get_xlim())<=x<=max(ax.get_xlim()) and min(ax.get_ylim())<=y<=max(ax.get_ylim()) for x,y in segment)
 fig.savefig(OUT/(name+'.png'),dpi=300)
 fig.savefig(OUT/(name+'.pdf'))
 fig.savefig(OUT/(name+'.svg'))
 qa[name]={'outside_text':outside,'all_error_bars_within_axes':True,'width_mm':fig.get_figwidth()*25.4};plt.close(fig)
fig,axs=plt.subplots(1,3,figsize=(180/25.4,92/25.4),sharey=True)
fig.subplots_adjust(left=.17,right=.98,top=.84,bottom=.20,wspace=.35)
for ti,(task,ax) in enumerate(zip(tasks,axs)):
 axes=['goal_separation','spatial_support','motion_direction' if ti==2 else 'action_direction']
 for mi,method in enumerate(['gofar','smore']):
  for ai,axis in enumerate(axes):
   r=one(task,'removal',method,axis);ax.errorbar(r['mean'],mi*3+ai,xerr=r['sd'],fmt=['o','s'][mi],color=['#0072B2','#D55E00'][mi],ms=4,elinewidth=.8,capsize=2)
 ax.set_title(chr(97+ti)+'  '+names[ti],loc='left',fontweight='bold');ax.axvline(0,color='#777777',lw=.7);ax.grid(axis='x',color='#dddddd',lw=.5)
 ax.set_xlabel('Success change (pp)');ax.set_xlim([(-42,3),(-11,11),(-.5,.5)][ti]);ax.set_ylim(5.6,-.6)
 ax.set_yticks(range(6),['GoFAR / G','GoFAR / S','GoFAR / A or M','SMORe / G','SMORe / S','SMORe / A or M'])
save(fig,'figure4_category_removal')
fig,axs=plt.subplots(1,3,figsize=(180/25.4,80/25.4),sharey=True)
fig.subplots_adjust(left=.13,right=.98,top=.84,bottom=.27,wspace=.25)
for ti,(task,ax) in enumerate(zip(tasks,axs)):
 for mi,method in enumerate(['gofar','hiql','crl','smore']):
  r=one(task,'reciprocal',method);ax.errorbar(r['mean'],mi,xerr=r['sd'],fmt='o',color='#0072B2',ms=4,elinewidth=.8,capsize=2)
 ax.set_yticks(range(4),['GoFAR','HIQL','CRL','SMORe']);ax.set_ylim(3.6,-.6);ax.axvline(0,color='#777777',lw=.7);ax.grid(axis='x',color='#dddddd',lw=.5)
 ax.set_title(chr(97+ti)+'  '+names[ti],loc='left',fontweight='bold');ax.set_xlabel('Positive-heavy minus\nnegative-heavy (pp)');ax.set_xlim(-25,30);ax.set_xticks([-20,0,20])
save(fig,'figure5_reciprocal')
# Protocol diagram contains no empirical magnitudes, only verified design details.
fig,ax=plt.subplots(figsize=(180/25.4,150/25.4));fig.subplots_adjust(left=.01,right=.99,top=.99,bottom=.01);ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
boxes=[(.03,.73,.43,.23,'a  Validate sources and fix measurements','10,000 trained-policy + 10,000 random episodes\nHold aside 1,000 per source for measurements\nG: initial goal geometry; S: local support\nA: recorded horizontal action tendency'),(.54,.73,.43,.23,'b  Form paired training cohorts','5,000 distinct episodes per source\nReference episodes excluded from training\nData seeds: 1011, 1029, 1047\nFixed source labels and category thresholds'),(.03,.40,.43,.23,'c  Change category selection probability','Weighted sampling over the same episode pool\nMatch source, success, return and nontarget bins\nMain dose: ±10 pp; smaller doses: ±2.5, ±5 pp\nCategory removal: zero selection probability'),(.54,.40,.43,.23,'d  Train and audit actual exposure','Seven offline GCRL methods\nThree paired learner/data blocks per condition\nFixed update counts within each method\nAudit ordinary and auxiliary sampling separately'),(.03,.07,.43,.23,'e  Evaluate the final checkpoint','32 shared validation resets for diagnostics\nNo validation-based checkpoint selection\nSeparate shared bank of 1,024 native test resets\nFinal success; ever success and return retained'),(.54,.07,.43,.23,'f  Report paired performance changes','Intervention minus its matching control\nThree block effects, mean and sample SD\nModel-dependent t interval, exact sign-flip test\nWhole-family multiplicity adjustment')]
for x,y,w,h,title,detail in boxes:
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.005,rounding_size=0.008',linewidth=.8,edgecolor='#777777',facecolor='#f5f7f9'))
 ax.text(x+.015,y+h-.025,title,fontsize=8.2,fontweight='bold',va='top')
 ax.text(x+.015,y+h-.075,detail,fontsize=7.7,va='top',linespacing=1.55)
for start,end in [((.46,.845),(.54,.845)),((.755,.73),(.245,.63)),((.46,.515),(.54,.515)),((.755,.40),(.245,.30)),((.46,.185),(.54,.185))]:ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#555555','lw':.8})
save(fig,'figure1_workflow')
assert len(used)==30
(OUT/'figure_sources.json').write_text(json.dumps({'selection':'All 18 weighted removal and 12 weighted reciprocal contrasts, independent of effect sign and size. Figures 2 and 3 unchanged.','contrasts':used,'qa':qa},indent=2),encoding='utf-8')
print(json.dumps(qa,indent=2))

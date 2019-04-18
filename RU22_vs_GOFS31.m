
%% User input

% Glider data location

% RU22
lon_lim = [120 134];
lat_lim = [30 40];
url_glider = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc';

model_name ='GOFS 3.1';

% GOFS3.1 outout model location
url_model = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';
var = 'temperature';

% Initial and final date
date_ini = ' ';  %if empty, date_ini is the firts time stamp in data
date_end = ' ';   %if empty, date_ini is the firts time stamp in data

fig ='no';

[timeg_vec,depthg_vec,tempg_matrix,timem,depthm,tempm] = ...
          glider_transect_model_comp(url_glider,model_name,url_model,var,fig,date_ini,date_end)      
      
     
%%      
siz_text = 20;
siz_title =20;

var_name = ncreadatt(url_glider,var,'ioos_category');
var_units = ncreadatt(url_glider,var,'units');

inst_id = ncreadatt(url_glider,'/','id');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

varg_matrix = tempg_matrix;

cc_vec = floor(min(min(varg_matrix))):1:ceil(max(max(varg_matrix)));
    
figure
set(gcf,'position',[327 434 1301 521*2])

subplot(211)
contourf(timeg_vec,-depthg_vec,varg_matrix,cc_vec,'.--k')
hold on
contour(timeg_vec,-depthg_vec,varg_matrix,[26 26],'-k','linewidth',2)
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along Track ',var_name,' Profile ',inst_name],'fontsize',siz_title)

cc = jet(length(cc_vec)-1);
colormap(cc)
c = colorbar;
c.Label.String = [var_name,' ','(',var_units,')'];
c.Label.FontSize = siz_text;
caxis([floor(min(min(varg_matrix))) ceil(max(max(varg_matrix)))])
set(c,'ytick',cc_vec)

tt_vec = unique(floor([timeg_vec(1),timeg_vec(1)+(timeg_vec(end)-timeg_vec(1))/10:(timeg_vec(end)-timeg_vec(1))/10:timeg_vec(end),timeg_vec(end)]));
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
xlim([tt_vec(1) timeg_vec(end)])

%ylim([-max(depthg_vec) 0])
%yticks(floor(-max(depthg_vec):max(depthg_vec)/5:0))
ylim([-100 0])
yticks(-100:20:0)
  
patch([datenum(2018,8,17,9,0,0) datenum(2018,8,17,12,0,0) datenum(2018,8,17,12,0,0) datenum(2018,8,17,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 
  
patch([datenum(2018,8,18,9,0,0) datenum(2018,8,18,12,0,0) datenum(2018,8,18,12,0,0) datenum(2018,8,18,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)  

patch([datenum(2018,8,19,9,0,0) datenum(2018,8,19,12,0,0) datenum(2018,8,19,12,0,0) datenum(2018,8,19,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,20,9,0,0) datenum(2018,8,20,12,0,0) datenum(2018,8,20,12,0,0) datenum(2018,8,20,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 

patch([datenum(2018,8,21,9,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,22,9,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 
  
patch([datenum(2018,8,23,9,0,0) datenum(2018,8,23,12,0,0) datenum(2018,8,23,12,0,0) datenum(2018,8,23,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,24,9,0,0) datenum(2018,8,24,12,0,0) datenum(2018,8,24,12,0,0) datenum(2018,8,24,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

varm = tempm;

subplot(212)
contourf(timem,-depthm,varm,cc_vec,'.--k')
hold on
contour(timem,-depthm,varm,[26 26],'-k','linewidth',2)
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along Track ',var_name,' Profile ',model_name],'fontsize',siz_title)

cc = jet(length(cc_vec)-1);
colormap(cc)
c = colorbar;
c.Label.String = [var_name,' ','(',var_units,')'];
c.Label.FontSize = siz_text;
caxis([floor(min(min(varg_matrix))) ceil(max(max(varg_matrix)))])
set(c,'ytick',cc_vec)

tt_vec = unique(floor([timeg_vec(1),timeg_vec(1)+(timeg_vec(end)-timeg_vec(1))/10:(timeg_vec(end)-timeg_vec(1))/10:timeg_vec(end),timeg_vec(end)]));
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
xlim([tt_vec(1) timeg_vec(end)])

%ylim([-max(depthg_vec) 0])
%yticks(floor(-max(depthg_vec):max(depthg_vec)/5:0))
ylim([-100 0])
yticks(-100:20:0)

patch([datenum(2018,8,17,9,0,0) datenum(2018,8,17,12,0,0) datenum(2018,8,17,12,0,0) datenum(2018,8,17,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 
  
patch([datenum(2018,8,18,9,0,0) datenum(2018,8,18,12,0,0) datenum(2018,8,18,12,0,0) datenum(2018,8,18,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)  

patch([datenum(2018,8,19,9,0,0) datenum(2018,8,19,12,0,0) datenum(2018,8,19,12,0,0) datenum(2018,8,19,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,20,9,0,0) datenum(2018,8,20,12,0,0) datenum(2018,8,20,12,0,0) datenum(2018,8,20,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 

patch([datenum(2018,8,21,9,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,22,9,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 
  
patch([datenum(2018,8,23,9,0,0) datenum(2018,8,23,12,0,0) datenum(2018,8,23,12,0,0) datenum(2018,8,23,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,24,9,0,0) datenum(2018,8,24,12,0,0) datenum(2018,8,24,12,0,0) datenum(2018,8,24,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;      
      
% Figure name
Fig_name = [folder,'Along_track_',var,'_prof_',inst_name,'_',datestr(timem(1),'mm-dd-yy'),'-',datestr(timem(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 


%% User input

% Glider data location

% RU22
lon_lim = [120 134];
lat_lim = [30 40];
url_glider = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc';

model_name ='GOFS 3.1';

% GOFS3.1 outout model location
url_model = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';
var_name = 'salinity';

% Initial and final date
date_ini = ' ';  %if empty, date_ini is the firts time stamp in data
date_end = ' ';   %if empty, date_ini is the firts time stamp in data

fig ='no';

[timeg_vec,depthg_vec,saltg_matrix,timem,depthm,salm] = ...
          glider_transect_model_comp(url_glider,model_name,url_model,var_name,fig,date_ini,date_end)      
      
%%      
siz_text = 20;
siz_title =20;

var_name = ncreadatt(url_glider,var,'ioos_category');
var_units = ncreadatt(url_glider,var,'units');

inst_id = ncreadatt(url_glider,'/','id');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

var_name = 'Salinity';
varg_matrix = saltg_matrix;

%cc_vec = floor(min(min(varg_matrix))):0.2:ceil(max(max(varg_matrix)));
cc_vec = 31:0.2:34;

figure
set(gcf,'position',[327 434 1301 521*2])

subplot(211)
contourf(timeg_vec,-depthg_vec,varg_matrix,cc_vec,'.--k')
hold on
contour(timeg_vec,-depthg_vec,varg_matrix,[26 26],'-k','linewidth',2)
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along Track ',var_name,' Profile ',inst_name],'fontsize',siz_title)

cc = jet(length(cc_vec)-1);
colormap(cc)
c = colorbar;
c.Label.String = [var_name,' ','(',var_units,')'];
c.Label.FontSize = siz_text;
%caxis([floor(min(min(varg_matrix))) ceil(max(max(varg_matrix)))])
caxis([31 34])
set(c,'ytick',cc_vec)

tt_vec = unique(floor([timeg_vec(1),timeg_vec(1)+(timeg_vec(end)-timeg_vec(1))/10:(timeg_vec(end)-timeg_vec(1))/10:timeg_vec(end),timeg_vec(end)]));
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
xlim([tt_vec(1) timeg_vec(end)])

%ylim([-max(depthg_vec) 0])
%yticks(floor(-max(depthg_vec):max(depthg_vec)/5:0))
ylim([-100 0])
yticks(-100:20:0)

patch([datenum(2018,8,17,9,0,0) datenum(2018,8,17,12,0,0) datenum(2018,8,17,12,0,0) datenum(2018,8,17,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 
  
patch([datenum(2018,8,18,9,0,0) datenum(2018,8,18,12,0,0) datenum(2018,8,18,12,0,0) datenum(2018,8,18,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)  

patch([datenum(2018,8,19,9,0,0) datenum(2018,8,19,12,0,0) datenum(2018,8,19,12,0,0) datenum(2018,8,19,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,20,9,0,0) datenum(2018,8,20,12,0,0) datenum(2018,8,20,12,0,0) datenum(2018,8,20,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 

patch([datenum(2018,8,21,9,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,22,9,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 
  
patch([datenum(2018,8,23,9,0,0) datenum(2018,8,23,12,0,0) datenum(2018,8,23,12,0,0) datenum(2018,8,23,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,24,9,0,0) datenum(2018,8,24,12,0,0) datenum(2018,8,24,12,0,0) datenum(2018,8,24,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;


varm = salm;

subplot(212)
contourf(timem,-depthm,varm,cc_vec,'.--k')
hold on
contour(timem,-depthm,varm,[26 26],'-k','linewidth',2)
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along Track ',var_name,' Profile ',model_name],'fontsize',siz_title)

cc = jet(length(cc_vec)-1);
colormap(cc)
c = colorbar;
c.Label.String = [var_name,' ','(',var_units,')'];
c.Label.FontSize = siz_text;
%caxis([floor(min(min(varg_matrix))) ceil(max(max(varg_matrix)))])
caxis([31 34])
set(c,'ytick',cc_vec)

tt_vec = unique(floor([timeg_vec(1),timeg_vec(1)+(timeg_vec(end)-timeg_vec(1))/10:(timeg_vec(end)-timeg_vec(1))/10:timeg_vec(end),timeg_vec(end)]));
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
xlim([tt_vec(1) timeg_vec(end)])

%ylim([-max(depthg_vec) 0])
%yticks(floor(-max(depthg_vec):max(depthg_vec)/5:0))
ylim([-100 0])
yticks(-100:20:0)

patch([datenum(2018,8,17,9,0,0) datenum(2018,8,17,12,0,0) datenum(2018,8,17,12,0,0) datenum(2018,8,17,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 
  
patch([datenum(2018,8,18,9,0,0) datenum(2018,8,18,12,0,0) datenum(2018,8,18,12,0,0) datenum(2018,8,18,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)  

patch([datenum(2018,8,19,9,0,0) datenum(2018,8,19,12,0,0) datenum(2018,8,19,12,0,0) datenum(2018,8,19,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,20,9,0,0) datenum(2018,8,20,12,0,0) datenum(2018,8,20,12,0,0) datenum(2018,8,20,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 

patch([datenum(2018,8,21,9,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,22,9,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5) 
  
patch([datenum(2018,8,23,9,0,0) datenum(2018,8,23,12,0,0) datenum(2018,8,23,12,0,0) datenum(2018,8,23,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)
  
patch([datenum(2018,8,24,9,0,0) datenum(2018,8,24,12,0,0) datenum(2018,8,24,12,0,0) datenum(2018,8,24,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.5)

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;  
      
% Figure name
Fig_name = [folder,'Along_track_',var_name,'_prof_',inst_name,'_',datestr(timem(1),'mm-dd-yy'),'-',datestr(timem(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300')

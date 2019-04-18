%% Glider track as a function of time

clear all;

%% User input

% ng300 (Virgin Islands)
lon_lim = [-68 -64];
lat_lim = [15 20];
gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng300-20180701T0000/ng300-20180701T0000.nc3.nc';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

%%

% ng300
url_glider = gdata;
model_name = 'GOFS 3.1';
url_model = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';
var = 'temperature';
fig = 'no';
date_ini = '04-Aug-2018 00:00:00';
date_end = '14-Aug-2018 00:00:00';

[timeg_vec, depthg_vec, varg_matrix,timem,depthm,varm] = ...
          glider_transect_model_comp(url_glider,model_name,url_model,var,fig,date_ini,date_end);

%%      
siz_text = 20;
siz_title =20;

var_name = ncreadatt(url_glider,var,'ioos_category');
var_units = ncreadatt(url_glider,var,'units');

inst_id = ncreadatt(url_glider,'/','id');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

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
title(['Along track ',var_name,' profile ',inst_name],'fontsize',siz_title)

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
ylim([-250 0])
yticks(-250:50:0)


%patch([datenum(2018,8,21,9,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,9,0,0)],...
%      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3) 
  
%patch([datenum(2018,8,22,9,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,9,0,0)],...
%      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3)   

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

%%
subplot(212)
contourf(timem,-depthm,varm,cc_vec,'.--k')
hold on
contour(timem,-depthm,varm,[26 26],'-k','linewidth',2)
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along track ',var_name,' profile ',model_name],'fontsize',siz_title)

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
ylim([-250 0])
yticks(-250:50:0)


%patch([datenum(2018,8,21,9,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,9,0,0)],...
%      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3) 
  
%patch([datenum(2018,8,22,9,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,9,0,0)],...
%      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3)  

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;      
      
% Figure name
Fig_name = [folder,'Along_track_',var,'_prof_',inst_name,'_',datestr(timem(1),'mm-dd-yy'),'-',datestr(timem(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%%

% ng300
url_glider = gdata;
model_name = 'GOFS 3.1';
url_model = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';
var = 'salinity';
fig = 'no';
date_ini = '04-Aug-2018 00:00:00';
date_end = '14-Aug-2018 00:00:00';

[timeg_vec, depthg_vec, varg_matrix,timem,depthm,varm] = ...
          glider_transect_model_comp(url_glider,model_name,url_model,var,fig,date_ini,date_end);

%%      
siz_text = 20;
siz_title =20;

var_name = ncreadatt(url_glider,var,'ioos_category');
var_units = ncreadatt(url_glider,var,'units');

cc_vec = floor(min(min(varg_matrix))):0.2:ceil(max(max(varg_matrix)));
    
figure
set(gcf,'position',[327 434 1301 521*2])

subplot(211)
contourf(timeg_vec,-depthg_vec,varg_matrix,cc_vec,'.--k')
hold on
contour(timeg_vec,-depthg_vec,varg_matrix,[26 26],'-k','linewidth',2)
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along track ',var_name,' profile ',inst_name],'fontsize',siz_title)

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
ylim([-250 0])
yticks(-250:50:0)


patch([datenum(2018,8,21,9,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3) 
  
patch([datenum(2018,8,22,9,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3)   

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

%%
subplot(212)
contourf(timem,-depthm,varm,cc_vec,'.--k')
hold on
contour(timem,-depthm,varm,[26 26],'-k','linewidth',2)
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along track ',var_name,' profile ',model_name],'fontsize',siz_title)

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
ylim([-250 0])
yticks(-250:50:0)


patch([datenum(2018,8,21,9,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,12,0,0) datenum(2018,8,21,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3) 
  
patch([datenum(2018,8,22,9,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,12,0,0) datenum(2018,8,22,9,0,0)],...
      [-300 -300 0 0],[0.5 0.5 0.5],'facealpha',0.3)  

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;      
      
% Figure name
Fig_name = [folder,'Along_track_',var,'_prof_',inst_name,'_',datestr(timem(1),'mm-dd-yy'),'-',datestr(timem(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

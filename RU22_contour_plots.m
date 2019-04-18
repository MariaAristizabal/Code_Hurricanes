
%% Glider track as a function of time

clear all;

%% User input

% Glider data location

% RU22
lon_lim = [120 134];
lat_lim = [30 40];
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

%%
% RU22
url_glider = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc';
var = 'temperature';
fig = 'no';
[timeg_vec, depthg_vec, varg_matrix] = glider_transect_contour(url_glider,var,fig);

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
set(gcf,'position',[1         461        1576         524])
contourf(timeg_vec,-depthg_vec,varg_matrix,cc_vec,'-','color',[0.5 0.5 0.5])
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

ylim([-max(depthg_vec) 0])
yticks(floor(-max(depthg_vec):max(depthg_vec)/5:0))

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

% Figure name
Fig_name = [folder,'Along_track_temp_prof_',inst_name,'_',datestr(timeg(1),'mm-dd-yy'),'-',datestr(timeg(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%%
url_glider = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc';
var = 'salinity';
fig = 'no';

[timeg_vec, depthg_vec, varg_matrix] = glider_transect_contour(url_glider,var,fig);

%%

siz_text = 20;
siz_title =20;

var_name = ncreadatt(url_glider,var,'ioos_category');
var_units = ncreadatt(url_glider,var,'units');

inst_id = ncreadatt(url_glider,'/','id');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

cc_vec = floor(min(min(varg_matrix))):0.5:ceil(max(max(varg_matrix)));

figure
set(gcf,'position',[1         461        1576         524])
contourf(timeg_vec,-depthg_vec,varg_matrix)%,cc_vec)%,'-','color',[0.5 0.5 0.5])
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

ylim([-max(depthg_vec) 0])
yticks(floor(-max(depthg_vec):max(depthg_vec)/5:0))

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

% Figure name
Fig_name = [folder,'Along_track_salt_prof_',inst_name,'_',datestr(timeg(1),'mm-dd-yy'),'-',datestr(timeg(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 


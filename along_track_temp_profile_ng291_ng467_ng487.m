%% Along track Temperature profiles 

clear all

%% User input

url_glider291 = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng291-20180701T0000/ng291-20180701T0000.nc3.nc';
url_glider467 = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc';
url_glider487 = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng487-20180701T0000/ng487-20180701T0000.nc3.nc';

% Inputs
var = 'temperature';
fig = 'no';
date_ini = '17-Jul-2018 00:00:00';
date_end = '17-Sep-2018 00:00:00';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';


%% glider_transect_contour

[timeg_vec291, depthg_vec291, tempg_matrix291] = ...
    glider_transect_contour(url_glider291,var,fig,date_ini,date_end);

[timeg_vec487, depthg_vec487, tempg_matrix487] = ...
    glider_transect_contour(url_glider487,var,fig,date_ini,date_end);

[timeg_vec467, depthg_vec467, tempg_matrix467] = ...
    glider_transect_contour(url_glider467,var,fig,date_ini,date_end);

%% Get lat and lon

%%%%%%%%%%
latit291 = double(ncread(url_glider291,'latitude'));
longi291 = double(ncread(url_glider291,'longitude'));

if ~exist('date_ini','var')
   tti = timeg_vec291(1);
else
   tti = datenum(date_ini); 
end

if ~exist('date_end','var')
   tte = timeg_vec291(end);
else
   tte = datenum(date_end); 
end

% Finding subset of data for time period of interest
ok_time_glider = find(timeg_vec291 >= tti & timeg_vec291 < tte);

lat291 = latit291(ok_time_glider);
lon291 = longi291(ok_time_glider);

%%%%%%%%%

latit487 = double(ncread(url_glider487,'latitude'));
longi487 = double(ncread(url_glider487,'longitude'));

if ~exist('date_ini','var')
   tti = timeg_vec487(1);
else
   tti = datenum(date_ini); 
end

if ~exist('date_end','var')
   tte = timeg_vec487(end);
else
   tte = datenum(date_end); 
end

% Finding subset of data for time period of interest
ok_time_glider = find(timeg_vec487 >= tti & timeg_vec487 < tte);

lat487 = latit487(ok_time_glider);
lon487 = longi487(ok_time_glider);

%%%%%%%%%
latit467 = double(ncread(url_glider467,'latitude'));
longi467 = double(ncread(url_glider467,'longitude'));

if ~exist('date_ini','var')
   tti = timeg_vec467(1);
else
   tti = datenum(date_ini); 
end

if ~exist('date_end','var')
   tte = timeg_vec467(end);
else
   tte = datenum(date_end); 
end

% Finding subset of data for time period of interest
ok_time_glider = find(timeg_vec467 >= tti & timeg_vec467 < tte);

lat467 = latit467(ok_time_glider);
lon467 = longi467(ok_time_glider);

%%
url_glider =url_glider291;
varg_matrix = tempg_matrix291;
timeg_vec = timeg_vec291;
depthg_vec = depthg_vec291;

inst_id = ncreadatt(url_glider,'/','id');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

siz_text = 20;
siz_title =20;

var_name = ncreadatt(url_glider,var,'ioos_category');
var_units = ncreadatt(url_glider,var,'units');

%cc_vec = floor(min(min(varg_matrix))):1:ceil(max(max(varg_matrix)));
cc_vec = 21:29;

figure
set(gcf,'position',[327 434 1301 521])
contourf(timeg_vec,-depthg_vec,varg_matrix,cc_vec,'.--k')
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along track ',var_name,' profile ',inst_name],'fontsize',siz_title)

cc = jet(length(cc_vec)-1);
colormap(cc)
c = colorbar;
c.Label.String = [var_name,' ','(',var_units,')'];
c.Label.FontSize = siz_text;
%caxis([floor(min(min(varg_matrix))) ceil(max(max(varg_matrix)))])
caxis([21 29])
set(c,'ytick',cc_vec)

%tt_vec = unique(floor([timeg_vec(1),timeg_vec(1)+(timeg_vec(end)-timeg_vec(1))/10:(timeg_vec(end)-timeg_vec(1))/10:timeg_vec(end),timeg_vec(end)]));
tt_vec = datenum(date_ini):7:datenum(date_end);
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
%xlim([tt_vec(1) timeg_vec(end)])
xlim([datenum(date_ini) datenum(date_end)])

%ylim([-max(depthg_vec) 0])
ylim([-200 0])
%yticks(floor(-max(depthg_vec):max(depthg_vec)/5:0))
yticks(-200:40:0)

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

% Figure name
Fig_name = [folder,'Along_track_temp_',inst_name,'_',datestr(timeg_vec(1),'mm-dd-yy'),'-',datestr(timeg_vec(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%%
url_glider =url_glider487;
varg_matrix = tempg_matrix487;
timeg_vec = timeg_vec487;
depthg_vec = depthg_vec487;

inst_id = ncreadatt(url_glider,'/','id');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

siz_text = 20;
siz_title =20;

var_name = ncreadatt(url_glider,var,'ioos_category');
var_units = ncreadatt(url_glider,var,'units');

%cc_vec = floor(min(min(varg_matrix))):1:ceil(max(max(varg_matrix)));
cc_vec = 21:29;

figure
set(gcf,'position',[327 434 1301 521])
contourf(timeg_vec,-depthg_vec,varg_matrix,cc_vec,'.--k')
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along track ',var_name,' profile ',inst_name],'fontsize',siz_title)

cc = jet(length(cc_vec)-1);
colormap(cc)
c = colorbar;
c.Label.String = [var_name,' ','(',var_units,')'];
c.Label.FontSize = siz_text;
%caxis([floor(min(min(varg_matrix))) ceil(max(max(varg_matrix)))])
caxis([21 29])
set(c,'ytick',cc_vec)

%tt_vec = unique(floor([timeg_vec(1),timeg_vec(1)+(timeg_vec(end)-timeg_vec(1))/10:(timeg_vec(end)-timeg_vec(1))/10:timeg_vec(end),timeg_vec(end)]));
tt_vec = datenum(date_ini):7:datenum(date_end);
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
%xlim([tt_vec(1) timeg_vec(end)])
xlim([datenum(date_ini) datenum(date_end)])

%ylim([-max(depthg_vec) 0])
ylim([-200 0])
%yticks(floor(-max(depthg_vec):max(depthg_vec)/5:0))
yticks(-200:40:0)

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

% Figure name
Fig_name = [folder,'Along_track_temp_',inst_name,'_',datestr(timeg_vec(1),'mm-dd-yy'),'-',datestr(timeg_vec(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%%
url_glider =url_glider467;
varg_matrix = tempg_matrix467;
timeg_vec = timeg_vec467;
depthg_vec = depthg_vec467;

inst_id = ncreadatt(url_glider,'/','id');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

siz_text = 20;
siz_title =20;

var_name = ncreadatt(url_glider,var,'ioos_category');
var_units = ncreadatt(url_glider,var,'units');

%cc_vec = floor(min(min(varg_matrix))):1:ceil(max(max(varg_matrix)));
cc_vec = 21:29;

figure
set(gcf,'position',[327 434 1301 521])
contourf(timeg_vec,-depthg_vec,varg_matrix,cc_vec,'.--k')
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along track ',var_name,' profile ',inst_name],'fontsize',siz_title)

cc = jet(length(cc_vec)-1);
colormap(cc)
c = colorbar;
c.Label.String = [var_name,' ','(',var_units,')'];
c.Label.FontSize = siz_text;
%caxis([floor(min(min(varg_matrix))) ceil(max(max(varg_matrix)))])
caxis([21 29])
set(c,'ytick',cc_vec)

%tt_vec = unique(floor([timeg_vec(1),timeg_vec(1)+(timeg_vec(end)-timeg_vec(1))/10:(timeg_vec(end)-timeg_vec(1))/10:timeg_vec(end),timeg_vec(end)]));
tt_vec = datenum(date_ini):7:datenum(date_end);
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
%xlim([tt_vec(1) timeg_vec(end)])
xlim([datenum(date_ini) datenum(date_end)])

%ylim([-max(depthg_vec) 0])
ylim([-200 0])
%yticks(floor(-max(depthg_vec):max(depthg_vec)/5:0))
yticks(-200:40:0)

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

% Figure name
Fig_name = [folder,'Along_track_temp_',inst_name,'_',datestr(timeg_vec(1),'mm-dd-yy'),'-',datestr(timeg_vec(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 
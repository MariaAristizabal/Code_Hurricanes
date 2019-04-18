
%% Along track Temperature profiles 

clear all

%% User input

% Pelagia (SAB)
%gdata = 'http://data.ioos.us//thredds/dodsC/deployments/secoora/pelagia-20180910T0000/pelagia-20180910T0000.nc3.nc';

% RU33 (MAB)
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';

% RAMSES (MAB)
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc';

% Silbo
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/silbo-20180525T1016/silbo-20180525T1016.nc3.nc';

% Initial and final date
date_ini = '11-Sep-2018 00:00:00';
%date_ini = '01-Aug-2018 00:00:00';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/';
%folder = 'htpps://marine.rutgers.edu/~aristizabal/MARACOOS_proj/Figures/';

%% Glider Extract

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');

temperature = double(ncread(gdata,'temperature'));
pressure = double(ncread(gdata,'pressure'));
time = double(ncread(gdata,'time'));
time = datenum(1970,01,01,0,0,time);
latitude = double(ncread(gdata,'latitude'));
longitude = double(ncread(gdata,'longitude'));

% Finding subset of data for time period of interest
tti = datenum(date_ini);
%tte = datenum(date_end);
ok_time_glider = find(time >= tti);

timeg = time(ok_time_glider);
tempg = temperature(:,ok_time_glider);
presg = pressure(:,ok_time_glider);

%% Figure

siz = 20;

inst = strsplit(inst_id,'-');
inst_name = inst{1};
plat = strsplit(plat_type,' ');

% Figure name
Fig_name = [folder,'Along_track_temp_prof_',inst_name,'_',plat{1},'_',plat{2},'_',date_ini(1:11)];

time_mat = repmat(timeg,1,size(tempg,1))';
time_vec = reshape(time_mat,1,size(time_mat,1)*size(time_mat,2));

depth_vec = reshape(presg,1,size(presg,1)*size(presg,2));
tempg_vec = reshape(tempg,1,size(tempg,1)*size(tempg,2));

marker.MarkerSize = 16;

figure
set(gcf,'position',[182 220 1286 574])

[h,c_h] = fast_scatter(time_vec',-depth_vec',tempg_vec','colorbar','vert','marker',marker);
c = colorbar;
datetick('x','mm/dd HH:MM','keepticks','keeplimits');
set(gca,'fontsize',siz)
ylabel('Depth (m)')
xlabel('Date')
c.Label.String = 'Potential Temperature (^oC)';
c.Label.FontSize = siz;
colormap('jet')
xlim([timeg(1) timeg(end)])
%title({'Along track temperature profile ' [inst_name,' ',plat_type]},'fontsize',siz)
title(['Along track temperature profile ',inst_name,' ',plat_type],'fontsize',siz)

wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 


%% Glider track map

clear all;

%% User input

% Bathymetry data
%bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

% Glider data location
% silbo
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/silbo-20180525T1016/silbo-20180525T1016.nc3.nc';

% RAMSES (MAB)
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc';

% Initial and final date
date_ini = '11-Sep-2018 00:00:00';
date_end = '12-Sep-2018 00:00:00';

% Region coordinates

% MAB
lon_lim = [-78 -72];
lat_lim = [34 40];

%MAB
%lon_lim = [-78 -70];
%lat_min = [35 43];

%Virgin Islands
%lon_lim = [-68 -64];
%lat_lim = [15 20];

% MAB + SAB
%lon_lim = [-81 -70];
%lat_lim = [30 42];

% Equatorial Atlantic
%lon_lim = [-35 -10];
%lat_lim = [12 32];

% Atlantic
%lon_lim = [-78 -64];
%lat_lim = [32 44];

% Golf of Mexico
%lon_lim = [-94 -78];
%lat_lim = [20 36];

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/';

%% Bathymetry data
%{
%disp(bath_data);
bath_lat = ncread(bath_data,'lat');
bath_lon = ncread(bath_data,'lon');
bath_elev = ncread(bath_data,'elevation');
%}
%% Glider Extract

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');

time = double(ncread(gdata,'time'));
time = datenum(1970,01,01,0,0,time);
latitude = double(ncread(gdata,'latitude'));
longitude = double(ncread(gdata,'longitude'));

% Finding subset of data for time period of interest
tti = datenum(date_ini);
tte = datenum(date_end);
ok_time_glider = find(time >= tti & time < tte);

% Final lat and lon of deployment for time period of interest 
% This lat and lon is used to find the grid point in the model output
% we can consider changing it to the average lat, lon for the entire
% deployment
Glat = latitude(ok_time_glider(end));
Glon = longitude(ok_time_glider(end));

%% Figure

filename = [folder,'Bathymetry_North_Atlantic.fig'];

openfig(filename)

xlim(lon_lim)
ylim(lat_lim)
hold on
plot(longitude,latitude,'.k')
pos=plot(Glon,Glat,'*r','markersize',12);
set(gca,'fontsize',16)
title([inst_name,' ',plat_type,' Track'],'fontsize',siz)
text(lon_lim(1)+(lon_lim(end)-lon_lim(1))*0.05,lat_lim(end)-(lat_lim(end)-lat_lim(1))*0.1,...
    ['Glider Position  {\color{red}{\ast}}',{num2str(Glon)},{num2str(Glat)}],...
    'fontsize',16,'backgroundcolor','w')

% Figure name
Fig_name = [folder,'Along_track_',inst_name,'_',plat_type,'_',date_ini(1:11)];

savefig(Fig_name)

%% Figure

%{
siz = 20;
mar_siz = 30;

figure
contourf(bath_lon,bath_lat,bath_elev')
hold on
contour(bath_lon,bath_lat,bath_elev',[0,-50,-100,-200,-1000,-2000,-4000,-8000],'color','k')
shading interp
axis equal
xlabel('Lon (^o)')
ylabel('Lat (^o)')
set(gca,'fontsize',16)

savefig('Bathymetry.fig')
%}

%% Figure

%{
% Instrument name:
inst = strsplit(inst_id,'-');
inst_name = inst{1};

siz = 20;
mar_siz = 30;

figure
contourf(bath_lon,bath_lat,bath_elev')
hold on
contour(bath_lon,bath_lat,bath_elev',[0,-50,-100,-200,-1000,-2000,-4000,-8000],'color','k')
shading interp
axis equal
xlim(lon_lim)
ylim(lat_lim)
hold on
plot(longitude,latitude,'.k')
pos=plot(Glon,Glat,'*r','markersize',12);
xlabel('Lon (^o)')
ylabel('Lat (^o)')
set(gca,'fontsize',16)
title([inst_name,' ',plat_type,' Track'],'fontsize',siz)
%title(['cp376',' ',plat_type,' Track'],'fontsize',siz)
%text(-77.5,42,['Glider Position  {\color{red}{\ast}}',{num2str(Glon)},{num2str(Glat)}],'fontsize',16)
%text(-67.5,20.5,['Glider Position  {\color{red}{\ast}}',{num2str(Glon)},{num2str(Glat)}],'fontsize',16)
text(lon_lim(1)+(lon_lim(end)-lon_lim(1))*0.05,lat_lim(end)-(lat_lim(end)-lat_lim(1))*0.1,...
    ['Glider Position  {\color{red}{\ast}}',{num2str(Glon)},{num2str(Glat)}],...
    'fontsize',16,'backgroundcolor','w')
%legend(pos,['Glider Position',{[num2str(Glon)]},{[num2str(Glat)]}],'location','northwest')

% Figure name
Fig_name = [folder,'Along_track_',inst_name,'_',plat_type,'_',date_ini(1:11)];

%savefig(Fig_name)

print([Fig_name,'.png'],'-dpng','-r300') 

%}
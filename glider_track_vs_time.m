%% Glider track as a function of time

%% User input

% Glider data location

% RU33 (MAB + SAB)
lon_lim = [-81 -70];
lat_lim = [30 42];
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';

% Initial and final date
%date_ini = '01-Sep-2018 00:00:00';
%date_end = '09-Sep-2018 00:00:00';
date_ini = ''; %if empty, date_ini is the firts time stamp in data
date_end = ''; %if empty, date_end is the last time stamp in data

% Bathymetry data
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

%% Glider Extract

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

time = double(ncread(gdata,'time'));
time = datenum(1970,01,01,0,0,time);
latitude = double(ncread(gdata,'latitude'));
longitude = double(ncread(gdata,'longitude'));

% Finding subset of data for time period of interest
if isempty(date_ini)
   tti = time(1);
else
   tti = datenum(date_ini);
end

if isempty(date_end)
   tte = time(end);
else
   tte = datenum(date_end);
end

ok_time_glider = find(time >= tti & time < tte);

% Finding subset of data for time period of interest
tti = datenum(tti);
tte = datenum(tte);
timeg = time(ok_time_glider);
latg = latitude(ok_time_glider);
long = longitude(ok_time_glider);

%% Bathymetry data

%disp(bath_data);
bath_lat = ncread(bath_data,'lat');
bath_lon = ncread(bath_data,'lon');
bath_elev = ncread(bath_data,'elevation');

oklatbath = bath_lat >= lat_lim(1) & bath_lat <= lat_lim(2);
oklonbath = bath_lon >= lon_lim(1) & bath_lon <= lon_lim(2);

bath_latsub = bath_lat(oklatbath);
bath_lonsub = bath_lon(oklonbath);
bath_elevsub = bath_elev(oklonbath,oklatbath);

%% Figure

siz_title = 20;
siz_text = 16;

figure
contour(bath_lonsub,bath_latsub,bath_elevsub',[0,-50,-100,-200,-1000,-2000,-4000,-8000],'color','k')
hold on
%colormap('summer')
contourf(bath_lonsub,bath_latsub,bath_elevsub')
shading interp
axis equal
xlim(lon_lim)
ylim(lat_lim)
xlabel('Lon (^o)')
ylabel('Lat (^o)')
set(gca,'fontsize',siz_text)
title([inst_name,' ',plat_type,' Track'],'fontsize',siz_title)
plot(long,latg,'.k','markersize',12)

%%
%{
%% Figure

siz_title = 20;
siz_text = 16;

color = colormap(jet(length(latitude)+1));
norm = (time-time(1))./(time(end)-time(1));
pos = round(norm.*length(latitude))+1;

figure

subplot(121)
contour(bath_lon,bath_lat,bath_elev',[0,-50,-100,-200,-1000,-2000,-4000,-8000],'color','k')
hold on
shading interp
axis equal
xlim(lon_lim)
ylim(lat_lim)
xlabel('Lon (^o)')
ylabel('Lat (^o)')
set(gca,'fontsize',siz_text)
title([inst_name,' ',plat_type,' Track'],'fontsize',siz_title)
for i=1:length(latitude)
    plot(longitude(i),latitude(i),'.','markersize',12,'color',color(pos(i),:,:))
end
%colormap('jet')
%c = colorbar('v');
%caxis([time(1) time(end)])
%time_vec = datenum(time(1)):datenum(0,0,2,0,0,0):datenum(time(end));
%time_lab = datestr(time_vec,'mmm/dd');
%set(c,'ytick',time_vec)
%set(c,'yticklabel',time_lab)

subplot(122)
contour(bath_lon,bath_lat,bath_elev',[0,-50,-100,-200,-1000,-2000,-4000,-8000],'color','k')
hold on
shading interp
axis equal
ylim([34 38])
xlim([-78 -74])
xlabel('Lon (^o)')
ylabel('Lat (^o)')
set(gca,'fontsize',siz_text)
title([inst_name,' ',plat_type,' Track'],'fontsize',siz_title)
for i=1:length(latitude)
    plot(longitude(i),latitude(i),'.','markersize',12,'color',color(pos(i),:,:))
end
colormap('jet')
c = colorbar('v');
caxis([time(1) time(end)])
time_vec = datenum(time(1)):datenum(0,0,2,0,0,0):datenum(time(end));
time_lab = datestr(time_vec,'mmm/dd');
set(c,'ytick',time_vec)
set(c,'yticklabel',time_lab)
%}
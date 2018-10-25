%% Glider track as a function of time

clear all;

%% User input

% Glider data location

% RAMSES (MAB + SAB)
lon_lim = [-81 -70];
lat_lim = [30 42];
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc';

% Initial and final date
%date_ini = '11-Sep-2018 00:00:00';
%date_end = '12-Sep-2018 00:00:00';

% Bathymetry data
%bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';
%filename = [folder,'Bathymetry_North_Atlantic.fig'];

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
%tti = datenum(date_ini);
%tte = datenum(date_end);
%timeg = time(ok_time_glider);
%tempg = temperature(:,ok_time_glider);
%presg = pressure(:,ok_time_glider); 

%% Bathymetry data

%disp(bath_data);
bath_lat = ncread(bath_data,'lat');
bath_lon = ncread(bath_data,'lon');
bath_elev = ncread(bath_data,'elevation');

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



%%
% Fake depth
%land = bath_elev > 0;
%ocean = bath_elev < 0;
%bath_elev(land) = datenum(2018,09,14,0,0,0);
%bath_elev(ocean) = datenum(2018,09,11,0,0,0);

%contourf(bath_lon,bath_lat,bath_elev')
%contourf(bath_lon,bath_lat,bath_elev',[-100,-8000])

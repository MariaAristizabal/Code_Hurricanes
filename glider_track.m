%% Glider track

%% User input

% Glider data location
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';
glider_name = 'RU33';

% Initial and final date
date_ini = '06-Sep-2018 00:00:00';
date_end = '07-Sep-2018 00:00:00';

% Figure name
Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/ru33_track_Aug_06';

%% Glider Extract

temperature = double(ncread(gdata,'temperature'));
pressure = double(ncread(gdata,'pressure'));
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


%% Bathymetry data

bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';

ncdisp(bath_data);
bath_lat = ncread(bath_data,'lat');
bath_lon = ncread(bath_data,'lon');
bath_elev = ncread(bath_data,'elevation');


%% Figure bathymetry

figure
contourf(bath_lon,bath_lat,bath_elev')
hold on
contour(bath_lon,bath_lat,bath_elev',[0,-50,-100,-200,-1000,-2000,-4000,-8000],'color','k')
shading interp
axis equal
xlim([-78 -70])
ylim([35 43])
hold on
plot(longitude,latitude,'.k')
plot(Glon,Glat,'*r','markersize',12)
xlabel('Lon (^o)')
ylabel('Lat (^o)')
set(gca,'fontsize',16)

%colorbar

%%

% R33
Glon = -72.9893;
Glat = 39.3673;

if Glon < 0 
    target_lon = 360 + Glon;
    lonGOFS = 360 + longitude;
else
    target_lon = Glon;
    lonGOFS = longitude;
end
target_lat = Glat;
latGOFS = latitude;


%% Figure glider track

figure 
pcolor(lon31,lat31,temp31')
xlim([282 292])
ylim([35 45])
shading interp
hold on
plot(lonGOFS,latGOFS,'.k')
plot(target_lon,target_lat,'*r','markersize',15)

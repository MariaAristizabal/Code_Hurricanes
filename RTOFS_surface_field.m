%% GOFS3.1 output model location

catalogRTOFS1 = 'http://nomads.ncep.noaa.gov:9090/dods/rtofs/rtofs_global20180913/rtofs_glo_3dz_nowcast_daily_temp';

%% GOFS 3.1

latRTOFS1 = ncread(catalogRTOFS1,'lat');
lonRTOFS1 = ncread(catalogRTOFS1,'lon');
tempRTOFS1 = squeeze(double(ncread(catalogRTOFS1,'temperature',[1 1 1 2],[inf inf 1 2])));

%% lat, lon convention

% R22
%Glon = 126.1363;
%Glat = 33.1150;

% R33
%Glon = -72.9893;
%Glat = 39.3673;

%ng300
%Glat = 17.6971;
%Glon = -65.0043;

% Ramses
Glon = -75.2601;
Glat = 35.4201;

if Glon < 0 
    target_lon = 360 + Glon;
else
    target_lon = Glon;
end
target_lat = Glat;

indRTOFS1 = find(lonRTOFS1 > target_lon);
oklonRTOFS1 = indRTOFS1(1);

indlatRTOFS1 = find(latRTOFS1 > target_lat);
oklatRTOFS1 = indlatRTOFS1(1);

figure 
pcolor(lonRTOFS1,latRTOFS1,tempRTOFS1(:,:,2)')
hold on
shading interp
plot(lonRTOFS1(oklonRTOFS1),latRTOFS1(oklatRTOFS1),'*k','markersize',10)

Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/RTOFS_lat_lon';
print([Fig_name,'.png'],'-dpng','-r300')

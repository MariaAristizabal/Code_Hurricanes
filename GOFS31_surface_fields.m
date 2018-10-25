
%% User input

date_ini = '20-Oct-2018 00:00:00';
date_end = '21-Oct-2018 00:00:00';

lat_min = 34;
lat_max = 38;
lon_min = -77 + 360;
lon_max = -74 + 360;

% GOFS3.1 output model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

%% GOFS 3.1

lat31 = ncread(catalog31,'lat');
lon31 = ncread(catalog31,'lon');
tim31 = ncread(catalog31,'time'); % hours since 2000-01-01 00:00:00
time31 = tim31/24 + datenum(2000,01,01,0,0,0);

tti = datenum(date_ini);
tte = datenum(date_end);
oktime31 = find(time31 >= tti & time31 < tte);

oklat31 = find(lat31 >= lat_min & lat31 <= lat_max);
oklon31 = find(lon31 >= lon_min & lon31 <= lon_max);

target_temp31 = squeeze(double(ncread(catalog31,'water_temp',[oklon31(1) oklat31(1) 1 oktime31(1)],[oklon31(end) oklat31(end) 1 oktime31(end)])));
target_salt31 = squeeze(double(ncread(catalog31,'salinity',[oklon31(1) oklat31(1) 1 oktime31(1)],[oklon31(end) oklat31(end) 1 oktime31(end)])));

%% lat, lon convention
%{
% R22
%Glon = 126.1363;
%Glat = 33.1150;

% R33
%Glon = -72.9893;
%Glat = 39.3673;

% ng300
Glon = -64.9920;
Glat = 17.7147;

if Glon < 0 
    target_lon = 360 + Glon;
else
    target_lon = Glon;
end
target_lat = Glat;

indlon = find(lon31 > target_lon);
oklon31 = indlon(1);

indlat31 = find(lat31 > target_lat);
oklat31 = indlat31(1);
%}

%% Temperature

figure 
pcolor(lon31,lat31,temp31')
hold on
shading interp
%plot(lon31(oklon31),lat31(oklat31),'*k','markersize',10)
colorbar

%Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Temperature/GOFS31_lat_lon';
%print([Fig_name,'.png'],'-dpng','-r300')

%% Salinity

figure 
pcolor(lon31,lat31,salt31')
hold on
shading interp
colorbar
caxis([32 36])

Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Salinity/GOFS31_salinity';
print([Fig_name,'.png'],'-dpng','-r300')

%% Salinity carebbean

figure 
pcolor(lon31,lat31,salt31')
hold on
shading interp
colorbar
caxis([34 37])
axis equal
xlim([260 360])
ylim([-5 40])

Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Salinity/GOFS31_salinity_detail';
print([Fig_name,'.png'],'-dpng','-r300')

%% Salinity carebbean

figure 
pcolor(lon31,lat31,salt31')
hold on
shading interp
colorbar
caxis([34 37])
axis equal
xlim([270 305])
ylim([10 26])

Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Salinity/GOFS31_salinity_detail2';
print([Fig_name,'.png'],'-dpng','-r300')

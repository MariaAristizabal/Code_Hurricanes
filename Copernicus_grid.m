%% Copernicus Surface fields

clear all;

% Initial and final date
date_ini = '11-Sep-2018 00:00:00';
date_end = '12-Sep-2018 00:00:00';

copern = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/global-analysis-forecast-phy-001-024_1537209157740.nc';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/';

%% 

latcop = ncread(copern,'latitude');
loncop = ncread(copern,'longitude');

timecop = ncread(copern,'time');
timecop = double(timecop)/24 + datenum('01-Jan-1950');

tti = datenum(date_ini);
tte = datenum(date_end);

okcop = find(timecop >= tti & timecop < tte);

saltcop = squeeze(double(ncread(copern,'so',[1 1 1 okcop(1)],[inf inf 1 1])));

%saltcop = squeeze(double(ncread(copern,'so',[okloncop oklarcop 1 okcop(1)],[1 1 1 1])));

%% lat, lon convention

% R22
%Glon = 126.1363;
%Glat = 33.1150;

% R33
%Glon = -72.9893;
%Glat = 39.3673;

% ng300
Glon = -73.3183;
Glat = 38.8307;

target_lon = Glon;
target_lat = Glat;

indlon = find(loncop > target_lon);
okloncop = indlon(1);

indlatcop = find(latcop > target_lat);
oklatcop = indlatcop(1);

figure 
pcolor(loncop,latcop,saltcop')
hold on
shading interp
plot(loncop(okloncop),latcop(oklatcop),'*k','markersize',10)
%plot(Glon,Glat,'*g','markersize',10)
colorbar
caxis([30 36])
colormap('jet')
%xlim([-76.5 -71])
%ylim([36 42])
title('Surface salinity from Copernicus on Sep 11 2018','fontsize',16)

Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Copern_lat_lon';
print([Fig_name,'.png'],'-dpng','-r300')
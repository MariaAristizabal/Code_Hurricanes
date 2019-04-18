%% Glider/Model Data Comparison

clear all;

%% User input

% Glider data location

% ng467 (Virgin Islands)
%lon_lim = [-68 -64];
%lat_lim = [15 20];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc';

% ng467 (Virgin Islands)
lon_lim = [-68 -64];
lat_lim = [15 20];
gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng302-20180701T0000/ng302-20180701T0000.nc3.nc';

% Golf of Mexico
%lon_lim = [-98 -78];
%lat_lim = [18 32];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

% ru22
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru22-20180815T0107/ru22-20180815T0107.nc3.nc';

% RU33 (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';

% RAMSES (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc';

% Pelagia (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/secoora/pelagia-20180910T0000/pelagia-20180910T0000.nc3.nc';

% ng300 (Virgin Islands)
%lon_lim = [-68 -64];
%lat_lim = [15 20];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng300-20180701T0000/ng300-20180701T0000.nc3.nc';

% sg630 (Virgin Islands)
%lon_lim = [-68 -64];
%lat_lim = [15 20];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/aoml/SG630-20180716T1220/SG630-20180716T1220.nc3.nc';

% silbo (Equatorial Atlantic)
%lon_lim = [-35 -10];
%lat_lim = [12 32];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/silbo-20180525T1016/silbo-20180525T1016.nc3.nc';

% blue (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/blue-20180806T1400/blue-20180806T1400.nc3.nc';

% cp367 West Atlantic
%lon_lim = [-78 -64];
%lat_lim = [32 44];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/cp_376-20180724T1552/cp_376-20180724T1552.nc3.nc';

% ng288 (Golf of Mexico)
%lon_lim = [-94 -78];
%lat_lim = [20 36];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

% ng342 (Golf of Mexico)
%lon_lim = [-94 -78];
%lat_lim = [20 36];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng342-20180701T0000/ng342-20180701T0000.nc3.nc';

% ng429 (Golf of Mexico)
%lon_lim = [-94 -78];
%lat_lim = [20 36];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng429-20180701T0000/ng429-20180701T0000.nc3.nc';

% Initial and final date
date_ini = '11-Sep-2018 00:00:00';
date_end = '12-Sep-2018 00:00:00';

% GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

% GOFS3.1 output model location
catalog30 = 'http://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_91.2/ts3z';

% Copernicus
%copern = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/global-analysis-forecast-phy-001-024_1536695141135.nc';
%copern = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/global-analysis-forecast-phy-001-024_1536948820036.nc';
copern = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/global-analysis-forecast-phy-001-024_1537209157740.nc';

% Bathymetry data
%bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';
%filename = [folder,'Bathymetry_North_Atlantic.fig'];

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';
%folder = 'htpps://marine.rutgers.edu/~aristizabal';
%folder = '/Volumes/coolgroup/Glider_models_comparisons/';

%% Glider Extract

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');

temperature = double(ncread(gdata,'temperature'));
salinity = double(ncread(gdata,'salinity'));
pressure = double(ncread(gdata,'pressure'));
time = double(ncread(gdata,'time'));
time = datenum(1970,01,01,0,0,time);
latitude = double(ncread(gdata,'latitude'));
longitude = double(ncread(gdata,'longitude'));

% Finding subset of data for time period of interest
tti = datenum(date_ini);
tte = datenum(date_end);
ok_time_glider = find(time >= tti & time < tte);

tempg = temperature(:,ok_time_glider);
presg = pressure(:,ok_time_glider);
latg = latitude(ok_time_glider);
long = longitude(ok_time_glider);
timeg = time(ok_time_glider);

% Final lat and lon of deployment for time period of interest 
% This lat and lon is used to find the grid point in the model output
% we can consider changing it to the average lat, lon for the entire
% deployment
Glat = latitude(ok_time_glider(end));
Glon = longitude(ok_time_glider(end));

%% Bathymetry data

%disp(bath_data);
bath_lat = ncread(bath_data,'lat');
bath_lon = ncread(bath_data,'lon');
bath_elev = ncread(bath_data,'elevation');

oklatbath = bath_lat >= lat_lim(1) & bath_lat <= lat_lim(2);
oklonbath = bath_lon >= lon_lim(1) & bath_lon <= lon_lim(2);

%% GOFS 3.1

%ncdisp(catalog31);

lat31 = ncread(catalog31,'lat');
lon31 = ncread(catalog31,'lon');
depth31 = ncread(catalog31,'depth');
tim31 = ncread(catalog31,'time'); % hours since 2000-01-01 00:00:00
time31 = tim31/24 + datenum(2000,01,01,0,0,0);
%oktime31 = find(time31 >= tti & time31 < tte);
oktime31 = find(time31 >= timeg(1) & time31 < timeg(end));

% Conversion from glider longitude and latitude to GOFS3.1 convention
target_lon(1:length(timeg)) = nan;
for i=1:length(timeg)
    if long(i) < 0 
       target_lon(i) = 360 + long(i);
    else
       target_lon(i) = long(i);
    end
end
target_lat = latg;

sublon31=interp1(timeg,target_lon,time31(oktime31));
sublat31=interp1(timeg,target_lat,time31(oktime31));

oklon31=round(interp1(lon31,1:length(lon31),sublon31));
oklat31=round(interp1(lat31,1:length(lat31),sublat31));

target_salt31(length(depth31),length(oktime31))=nan;
target_temp31(length(depth31),length(oktime31))=nan;
for i=1:length(oklon31)
    disp(length(oklon31))
    disp(['GOFS31',' i= ',num2str(i)])
    if isfinite(oklon31(i)) && isfinite(oklat31(i)) && isfinite(oktime31(i))
       target_temp31(:,i) = squeeze(double(ncread(catalog31,'water_temp',[oklon31(i) oklat31(i) 1 oktime31(i)],[1 1 inf 1])));
       target_salt31(:,i) = squeeze(double(ncread(catalog31,'salinity',[oklon31(i) oklat31(i) 1 oktime31(i)],[1 1 inf 1])));
    else
       target_temp31(:,i) = nan;
       target_salt31(:,i) = nan;
    end
end

%oklon31=round(interp1(lon31,1:length(lon31),target_lon));
%oklat31=round(interp1(lat31,1:length(lat31),target_lat));
%ok31 = find(time_matlab31 >= tti & time_matlab31 < tte);

%target_temp31 = squeeze(double(ncread(catalog31,'water_temp',[oklon31 oklat31 1 ok31(1)],[1 1 inf length(ok31)])));
%target_salt31 = squeeze(double(ncread(catalog31,'salinity',[oklon31 oklat31 1 ok31(1)],[1 1 inf length(ok31)])));

%% GOFS 3.0
%{
lat30 = ncread(catalog30,'lat');
lon30 = ncread(catalog30,'lon');
depth30 = ncread(catalog30,'depth');
time30 = ncread(catalog30,'time'); % hours since 2000-01-01 00:00:00

time_matlab30 = time30/24 + datenum(2000,01,01,0,0,0);
ok30 = find(time_matlab30 >= tti & time_matlab30 < tte);

time30 = ncread(catalog30,'time'); % hours since 2000-01-01 00:00:00
time30 = time30/24 + datenum(2000,01,01,0,0,0);
oktime30 = find(time30 >= tti & time30 < tte);
%}
%{
sublon30=interp1(timeg,target_lon,time30(oktime30));
sublat30=interp1(timeg,target_lat,time30(oktime30));

oklon30=round(interp1(lon30,1:length(lon30),sublon30));
oklat30=round(interp1(lat30,1:length(lat30),sublat30));

target_temp30(length(depth30),length(oktime30))=nan;
target_salt30(length(depth30),length(oktime30))=nan;
for i=1:length(oklon30)
    disp(length(oklon30))
    disp(['GOFS30',' i= ',num2str(i)])
    if isfinite(oklon30(i)) && isfinite(oklat30(i)) && isfinite(oktime30(i))
       target_temp30(:,i) = squeeze(double(ncread(catalog30,'water_temp',[oklon30(i) oklat30(i) 1 oktime30(i)],[1 1 inf 1])));
       target_salt30(:,i) = squeeze(double(ncread(catalog30,'salinity',[oklon30(i) oklat30(i) 1 oktime30(i)],[1 1 inf 1])));
    end
end
%}

%{
% GOFS 3.0 output is daily
oklon30=round(interp1(lon30,1:length(lon30),mean(target_lon)));
oklat30=round(interp1(lat30,1:length(lat30),mean(target_lat)));

target_temp30 = squeeze(double(ncread(catalog30,'water_temp',[oklon30 oklat30 1 ok30(1)],[1 1 inf length(ok30)])));
target_salt30 = squeeze(double(ncread(catalog30,'salinity',[oklon30 oklat30 1 ok30(1)],[1 1 inf length(ok30)])));
%}

%% RTOFS Global

%{
catalogRTOFS1 = 'http://nomads.ncep.noaa.gov:9090/dods/rtofs/rtofs_global20180913/rtofs_glo_3dz_nowcast_daily_temp';

latRTOFS1 = ncread(catalogRTOFS1,'lat');
lonRTOFS1 = ncread(catalogRTOFS1,'lon');
depthRTOFS1 = ncread(catalogRTOFS1,'lev');
timeRT = ncread(catalogRTOFS1,'time'); % hours since ??????

timeRTOFS1 = timeRT + datenum(1,0,0,0,0,0) - datenum(0,1,1,0,0,0) ;
%datestr(timeRT(1)+datenum(1,0,0,0,0,0)-datenum(0,1,1,0,0,0),0)

% Conversion from glider longitude and latitude to RTOFS global convention
if Glon < 0 
    target_lon = 360 + Glon;
else
    target_lon = Glon;
end
target_lat = Glat;

%indlon = find(lonRTOFS1 > target_lon);
%oklonRTOFS1 = indlon(1);
oklatRTOFS1=round(interp1(latRTOFS1,1:length(latRTOFS1),target_lon));

%indlat = find(latRTOFS1 > target_lat);
%oklatRTOFS1 = indlat(1);
oklonRTOFS1=round(interp1(lonRTOFS1,1:length(lonRTOFS1),target_lon));

okRTOFS1 = find(timeRTOFS1 >= tti & timeRTOFS1 < tte);

target_tempRTOFS1 = squeeze(double(ncread(catalogRTOFS1,'temperature',[oklonRTOFS1 oklatRTOFS1 1 okRTOFS1(1)],[1 1 inf length(okRTOFS1)])));
%}

%% RTOFS Regional
%{
catalogRTOFS2 = 'http://nomads.ncep.noaa.gov:9090/dods/rtofs/rtofs_global20180906/rtofs_glo_3dz_nowcast_6hrly_us_east';

latRTOFS1 = ncread(catalogRTOFS2,'lat');
lonRTOFS = ncread(catalogRTOFS2,'lon');
depthRTOFS1 = ncread(catalogRTOFS2,'lev');
timeRT = ncread(catalogRTOFS2,'time'); % hours since ??????

timeRTOFS1 = timeRT + datenum(1,0,0,0,0,0);

% Conversion from glider longitude and latitude to RTOFS regional convention
target_lon =  Glon;
target_lat = Glat;

indlon = find(lonRTOFS > target_lon);
oklonRTOFS1 = indlon(1);

indlat = find(latRTOFS1 > target_lat);
oklatRTOFS1 = indlat(1);

okRTOFS2 = find(timeRTOFS1 >= tti & timeRTOFS1 < tte);

target_tempRTOFS1 = squeeze(double(ncread(catalogRTOFS2,'temperature',[oklonRTOFS1 oklatRTOFS1 1 okRTOFS2(1)],[1 1 inf length(okRTOFS2)])));
%}

%% Copernicus
%{
%ncdisp(copern);

latcop = ncread(copern,'latitude');
loncop = ncread(copern,'longitude');
depthcop = ncread(copern,'depth');

timecop = ncread(copern,'time');
timecop = double(timecop)/24 + datenum('01-Jan-1950');

okcop = find(timecop >= tti & timecop < tte);

%indloncop = find(loncop > Glon);
%okloncop = indloncop(1);
okloncop=round(interp1(loncop,1:length(loncop),Glon));

%indlatcop = find(latcop > Glat);
%oklatcop = indlatcop(1);
oklatcop=round(interp1(latcop,1:length(latcop),Glat));

target_tempcop = squeeze(double(ncread(copern,'thetao',[okloncop oklatcop 1 okcop(1)],[1 1 inf length(okcop)])));
target_saltcop = squeeze(double(ncread(copern,'so',[okloncop oklatcop 1 okcop(1)],[1 1 inf length(okcop)])));
%}

%% Temporal frequency

% Glider
freq_glider = diff(time)*24*60;

% GOFS3.1 
freq_GOFS31 = (time31(2)-time31(1))*24;  %3 hourly

% GOFS3.0
%freq_GOFS30 = (time30(2)-time30(1))*24; %daily

% Copernicus
%freq_Coper = (timecop(2)-timecop(1))*24; %daily

%% Mean profiles

depth_2d = repmat(depth31,[1,length(oktime31)]);
%depth30_2d = repmat(depth30,[1,length(oktime30)]);
%depthcop_2d = repmat(depthcop,[1,length(okcop)]);
%depthRTOFS1_2d = repmat(depthRTOFS1,[1,length(okRTOFS1)]);

temp31_mean = mean(target_temp31,2);
%temp30_mean = mean(target_temp30,2);
%tempcop_mean = mean(target_tempcop,2);
%tempRTOFS1_mean = mean(target_tempRTOFS1,2);

salt31_mean = mean(target_salt31,2);
%salt30_mean = mean(target_salt30,2);
%saltcop_mean = mean(target_saltcop,2);
%saltRTOFS1_mean = mean(target_saltRTOFS1,2);

pres_gridded = 0:0.5:max(max(pressure(:,ok_time_glider)));

presok = pressure(:,ok_time_glider);
tempok = temperature(:,ok_time_glider);
temp_gridded(length(pres_gridded),size(presok,2)) = nan;

saltok = salinity(:,ok_time_glider);
salt_gridded(length(pres_gridded),size(presok,2)) = nan;

for i=1:size(pressure(:,ok_time_glider),2)
    [presu,oku] = unique(presok(:,i));
    tempu = tempok(oku,i);
    saltu = saltok(oku,i);
    ok = isfinite(presu);
    %ok = isfinite(tempu);
    temp_gridded(:,i) = interp1(presu(ok),tempu(ok),pres_gridded);
    salt_gridded(:,i) = interp1(presu(ok),saltu(ok),pres_gridded);
end

tempgl_mean = nanmean(temp_gridded,2);
saltgl_mean = nanmean(salt_gridded,2);


%% Figure temperature

% Instrument name:
inst = strsplit(inst_id,'-');
inst_name = inst{1};

%plat = strsplit(plat_type,' ');

siz = 20;
mar_siz = 30;
ok26 = find(tempgl_mean >= 26.0 & tempgl_mean < 26.2); 

figure
set(gcf,'position',[139 144 1157 811])

subplot(121)
plot(temperature(:,ok_time_glider),-pressure(:,ok_time_glider),'.-c','markersize',mar_siz)
hold on
plot(target_temp31,-depth_2d,'.-r','markersize',mar_siz)
%plot(target_temp30,-depth30_2d,'.-r','markersize',mar_siz)
%plot(target_tempcop,-depthcop_2d,'.-','color',[1 0.5 0],'markersize',mar_siz)
%plot(target_tempRTOFS1,-depthRTOFS1_2d,'.-','color',[1 0.5 1],'markersize',mar_siz)
h1 = plot(tempgl_mean,-pres_gridded,'.-b','markersize',mar_siz,'linewidth',4);
h2 = plot(temp31_mean,-depth31,'.-r','markersize',mar_siz,'linewidth',4);
if ~isempty(ok26)
%plot(26.0,-pres_gridded(ok26(1)),'^r','markersize',10,'markerfacecolor','r')
dd1 = -max(pres_gridded):0;
tt1 = repmat(26.0,1,length(dd1));
plot(tt1,dd1,'--k')
tt2 = 15:26;
dd2 = repmat(-pres_gridded(ok26(1)),1,length(tt2));
%plot(tt2,dd2,'--k')
end
%h3 = plot(temp30_mean,-depth30,'.-m','markersize',mar_siz,'linewidth',4);
%h4 = plot(tempcop_mean,-depthcop,'.-','color',[1 0.5 0],'markersize',mar_siz,'linewidth',4);
%h5 = plot(tempRTOFS1_mean,-depthRTOFS1,'.-','color',[1 0 0.2],'markersize',mar_siz,'linewidth',4);
set(gca,'fontsize',siz)
%lgd = legend([h1 h2 h3 h4],['cp376',' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
%lgd = legend([h1 h2 h3 h4],[inst_name,' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
%    ['HYCOM GOFS 3.1 Expt 93.0 (hindcast) ' datestr(time_matlab31(ok31(1)))],...
%    ['HYCOM GOFS 3.0 Expt 91.2 (hindcast) ' datestr(time_matlab30(ok30(1)))],...
%    'Location','SouthEast');

%lgd = legend([h1 h2 h3],[inst_name,' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
%    ['HYCOM GOFS 3.1 Expt 93.0 (hindcast) ' datestr(time31(oktime31(1)))],...
%    ['HYCOM GOFS 3.0 Expt 91.2 (hindcast) ' datestr(time30(oktime30(1)))],...
%    'Location','SouthEast');

lgd = legend([h1 h2],[inst_name,' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
    ['HYCOM GOFS 3.1 Expt 93.0 (hindcast) ' datestr(time31(oktime31(1)))],...
    'Location','SouthEast');

%    ['Copernicus' datestr(timecop(okcop(1)))],...
%    'Location','SouthEast');
set(lgd,'fontsize',14)
title({'HYCOM GOFS 3.1 and ';[inst_name,' ',plat_type]},'fontsize',siz)
%title({'HYCOM GOFS 3.1 and 3.0 and Copernicus vs ';['cp376',' ',plat_type]},'fontsize',siz)
xlabel('Temperature (^oC)','fontsize',siz)
ylabel('Depth (m)','fontsize',siz);
grid on;
if max(max(pressure(:,ok_time_glider))) > 200
    ylim([-(200+200*0.2) 0])
else
    ylim([-max(max(pressure(:,ok_time_glider)))-max(max(pressure(:,ok_time_glider)))*0.2 0])
end



s2 = subplot(122);
contourf(bath_lon(oklonbath),bath_lat(oklatbath),bath_elev(oklonbath,oklatbath)')
hold on
contour(bath_lon(oklonbath),bath_lat(oklatbath),bath_elev(oklonbath,oklatbath)',[0,-50,-100,-200,-1000,-2000,-4000,-8000],'color','k')
shading interp
axis equal
%h2 = openfig(filename,'reuse');
%ax2 = gca;
%fig2 =get(ax2,'children');
%copyobj(fig2,s2);
xlim(lon_lim)
ylim(lat_lim)
hold on
plot(longitude,latitude,'.k')
plot(Glon,Glat,'*r','markersize',12);
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
caxis([min(min(bath_elev)) max(max(bath_elev))])
%legend(pos,['Glider Position',{[num2str(Glon)]},{[num2str(Glat)]}],'location','northwest')

% Figure name
Fig_name = [folder,'models_vs_',inst_name,'_',date_ini(1:11)];

print([Fig_name,'_temp.png'],'-dpng','-r300') 
%print([Fig_name,'.eps'],'-depsc','-r300') 


%% Figure salinity

% Instrument name:
inst = strsplit(inst_id,'-');
inst_name = inst{1};

%plat = strsplit(plat_type,' ');

siz = 20;
mar_siz = 30;

figure
set(gcf,'position',[139 144 1157 811])

subplot(121)
plot(salinity(:,ok_time_glider),-pressure(:,ok_time_glider),'.-c','markersize',mar_siz)
hold on
plot(target_salt31,-depth_2d,'.-r','markersize',mar_siz)
%plot(target_salt30,-depth30_2d,'.-r','markersize',mar_siz)
%plot(target_saltcop,-depthcop_2d,'.-','color',[1 0.5 0],'markersize',mar_siz)
%plot(target_tempRTOFS1,-depthRTOFS1_2d,'.-','color',[1 0.5 1],'markersize',mar_siz)
h1 = plot(saltgl_mean,-pres_gridded,'.-b','markersize',mar_siz,'linewidth',4);
h2 = plot(salt31_mean,-depth31,'.-r','markersize',mar_siz,'linewidth',4);
%h3 = plot(salt30_mean,-depth30,'.-m','markersize',mar_siz,'linewidth',4);
%h4 = plot(saltcop_mean,-depthcop,'.-','color',[1 0.5 0],'markersize',mar_siz,'linewidth',4);
%h5 = plot(tempRTOFS1_mean,-depthRTOFS1,'.-','color',[1 0 0.2],'markersize',mar_siz,'linewidth',4);
set(gca,'fontsize',siz)
%lgd = legend([h1 h2 h3 h4],['cp376',' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
%lgd = legend([h1 h2 h3 h4],[inst_name,' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
%    ['HYCOM GOFS 3.1 Expt 93.0 (hindcast) ' datestr(time31(ok31(1)))],...
%    ['HYCOM GOFS 3.0 Expt 91.2 (hindcast) ' datestr(time_30(ok30(1)))],...
%    ['Copernicus' datestr(timecop(okcop(1)))],...
%    'Location','SouthEast');
%lgd = legend([h1 h2 h3],[inst_name,' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
%    ['HYCOM GOFS 3.1 Expt 93.0 (hindcast) ' datestr(time31(oktime31(1)))],...
%    ['HYCOM GOFS 3.0 Expt 91.2 (hindcast) ' datestr(time30(oktime30(1)))],...
%    'Location','SouthEast');
lgd = legend([h1 h2],[inst_name,' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
    ['HYCOM GOFS 3.1 Expt 93.0 (hindcast) ' datestr(time31(oktime31(1)))],...
    'Location','SouthEast');
set(lgd,'fontsize',14)
title({'HYCOM GOFS 3.1 vs ';[inst_name,' ',plat_type]},'fontsize',siz)
%title({'HYCOM GOFS 3.1 and 3.0 and Copernicus vs ';['cp376',' ',plat_type]},'fontsize',siz)
xlabel('Salinity (psu)','fontsize',siz)
ylabel('Depth (m)','fontsize',siz);
grid on;
if max(max(pressure(:,ok_time_glider))) > 200
    ylim([-(200+200*0.2) 0])
else
    ylim([-max(max(pressure(:,ok_time_glider)))-max(max(pressure(:,ok_time_glider)))*0.2 0])
end

subplot(122)
contourf(bath_lon(oklonbath),bath_lat(oklatbath),bath_elev(oklonbath,oklatbath)')
hold on
contour(bath_lon(oklonbath),bath_lat(oklatbath),bath_elev(oklonbath,oklatbath)',[0,-50,-100,-200,-1000,-2000,-4000,-8000],'color','k')
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
caxis([min(min(bath_elev)) max(max(bath_elev))])

% Figure name
Fig_name = [folder,'models_vs_',inst_name,'_',date_ini(1:11)];

print([Fig_name,'_salt.png'],'-dpng','-r300') 
%print([Fig_name,'.eps'],'-depsc','-r300') 

%% Figure temperature including RTOFOS
%{
% Instrument name:
inst = strsplit(inst_id,'-');
inst_name = inst{1};

%plat = strsplit(plat_type,' ');

siz = 20;
mar_siz = 30;

figure
set(gcf,'position',[139 144 1157 811])

subplot(121)
plot(temperature(:,ok_time_glider),-pressure(:,ok_time_glider),'.-g','markersize',mar_siz)
hold on
plot(target_temp31,-depth_2d,'.-c','markersize',mar_siz)
plot(target_temp30,-depth30_2d,'.-r','markersize',mar_siz)
plot(target_tempcop,-depthcop_2d,'.-','color',[1 0.5 0],'markersize',mar_siz)
plot(target_tempRTOFS1,-depthRTOFS1_2d,'.-','color',[1 0.5 1],'markersize',mar_siz)
h1 = plot(tempgl_mean,-pres_gridded,'.-k','markersize',mar_siz,'linewidth',4);
h2 = plot(temp31_mean,-depth,'.-b','markersize',mar_siz,'linewidth',4);
h3 = plot(temp30_mean,-depth30,'.-m','markersize',mar_siz,'linewidth',4);
h4 = plot(tempcop_mean,-depthcop,'.-','color',[1 0.5 0],'markersize',mar_siz,'linewidth',4);
h5 = plot(tempRTOFS1_mean,-depthRTOFS1,'.-','color',[1 0 0.2],'markersize',mar_siz,'linewidth',4);
set(gca,'fontsize',siz)
lgd = legend([h1 h2 h3 h4 h5],[inst_name,' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
    ['HYCOM GOFS 3.1 Expt 93.0 (hindcast) ' datestr(time_matlab31(ok31(1)))],...
    ['HYCOM GOFS 3.0 Expt 91.2 (hindcast) ' datestr(time_matlab30(ok30(1)))],...
    ['Copernicus' datestr(timecop(okcop(1)))],...
    ['RTOFS ' datestr(timeRTOFS1(okRTOFS1(1)))],...
    'Location','SouthEast');
set(lgd,'fontsize',14)
title({'HYCOM GOFS 3.1 and 3.0, Copernicus and RTOFS vs ';[inst_name,' ',plat_type]},'fontsize',siz)
xlabel('Temperature (^oC)','fontsize',siz)
ylabel('Depth (m)','fontsize',siz);
grid on;
if max(max(pressure(:,ok_time_glider))) > 200
    ylim([-(200+200*0.2) 0])
else
    ylim([-max(max(pressure(:,ok_time_glider)))-max(max(pressure(:,ok_time_glider)))*0.2 0])
end


subplot(122)
contourf(bath_lon,bath_lat,bath_elev')
hold on
contour(bath_lon,bath_lat,bath_elev',[0,-50,-100,-200,-1000,-2000,-4000,-8000],'color','k')
shading interp
axis equal
%MAB
xlim(lon_lim)
ylim(lat_lim)
hold on
plot(longitude,latitude,'.k')
pos=plot(Glon,Glat,'*r','markersize',12);
xlabel('Lon (^o)')
ylabel('Lat (^o)')
set(gca,'fontsize',16)
title([inst_name,' ',plat_type,' Track'],'fontsize',siz)
%text(-77.5,42,['Glider Position  {\color{red}{\ast}}',{num2str(Glon)},{num2str(Glat)}],'fontsize',16)
%text(-67.5,20.5,['Glider Position  {\color{red}{\ast}}',{num2str(Glon)},{num2str(Glat)}],'fontsize',16)
text(lon_lim(1)+(lon_lim(end)-lon_lim(1))*0.05,lat_lim(end)-(lat_lim(end)-lat_lim(1))*0.1,...
    ['Glider Position  {\color{red}{\ast}}',{num2str(Glon)},{num2str(Glat)}],...
    'fontsize',16,'backgroundcolor','w')
%legend(pos,['Glider Position',{[num2str(Glon)]},{[num2str(Glat)]}],'location','northwest')

% Figure name
Fig_name = [folder,'models_vs_',inst_name,'_',plat_type,'_',date_ini(1:11)];

print([Fig_name,'.png'],'-dpng','-r300') 
%}
clear all;

 %% User input

% Initial and final date
date_ini = '11-Sep-2018 00:00:00';
date_end = '12-Sep-2018 00:00:00';

% GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

% GOFS3.1 output model location
%catalog30 = 'http://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_91.2/ts3z';

% Bathymetry data
%bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

% Copernicus
%copern = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/global-analysis-forecast-phy-001-024_1537209157740.nc';
%copern = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/global-analysis-forecast-phy-001-024_1538489276306.nc';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';
%folder = '/Volumes/aristizabal/public_html/MARACOOS_proj/Figures/Model_glider_comp/';

% Glider data location
url = 'https://data.ioos.us/thredds/dodsC/deployments/';

%{
% Gulf Mexico
lon_lim = [-98 -78];
lat_lim = [18 32];
id_list = char({'rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc',...
           'rutgers/ng261-20180801T0000/ng261-20180801T0000.nc3.nc',...
           'rutgers/ng257-20180801T0000/ng257-20180801T0000.nc3.nc',...
           'rutgers/ng290-20180701T0000/ng290-20180701T0000.nc3.nc',...
           'rutgers/ng230-20180801T0000/ng230-20180801T0000.nc3.nc',...
           'rutgers/ng279-20180801T0000/ng279-20180801T0000.nc3.nc',...
           'rutgers/ng429-20180701T0000/ng429-20180701T0000.nc3.nc',...
           'secoora/sam-20180824T0000/sam-20180824T0000.nc3.nc',...
           'gcoos_dmac/Sverdrup-20180509T1742/Sverdrup-20180509T1742.nc3.nc',...
           'rutgers/ng258-20180801T0000/ng258-20180801T0000.nc3.nc',...
           'rutgers/ng295-20180701T0000/ng295-20180701T0000.nc3.nc',...
           'rutgers/ng296-20180701T0000/ng296-20180701T0000.nc3.nc',...
           'rutgers/ng228-20180801T0000/ng228-20180801T0000.nc3.nc',...
           'rutgers/ng309-20180701T0000/ng309-20180701T0000.nc3.nc',...
           'rutgers/ng342-20180701T0000/ng342-20180701T0000.nc3.nc',...
           'rutgers/ng448-20180701T0000/ng448-20180701T0000.nc3.nc',...
           'rutgers/ng450-20180701T0000/ng450-20180701T0000.nc3.nc',...
           'rutgers/ng464-20180701T0000/ng464-20180701T0000.nc3.nc',...
           'rutgers/ng466-20180701T0000/ng466-20180701T0000.nc3.nc',...
           'rutgers/ng489-20180701T0000/ng489-20180701T0000.nc3.nc',...
           'rutgers/ng512-20180701T0000/ng512-20180701T0000.nc3.nc',...
           'rutgers/ng596-20180701T0000/ng596-20180701T0000.nc3.nc',...
           'gcoos_dmac/Reveille-20180627T1500/Reveille-20180627T1500.nc3/nc',...
           }); 
%}


%{
%Caribbean:
lon_lim = [-68 -64];
lat_lim = [15 20];
id_list = char({'aoml/SG630-20180716T1220/SG630-20180716T1220.nc3.nc',...
                'aoml/SG610-20180719T1146/SG610-20180719T1146.nc3.nc',...
                'aoml/SG635-20180716T1248/SG635-20180716T1248.nc3.nc',...
                'aoml/SG649-20180731T1418/SG649-20180731T1418.nc3.nc',...
                'rutgers/ng300-20180701T0000/ng300-20180701T0000.nc3.nc',...
                'rutgers/ng291-20180701T0000/ng291-20180701T0000.nc3.nc',...
                'rutgers/ng302-20180701T0000/ng302-20180701T0000.nc3.nc',...
                'rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc',...
                'rutgers/ng487-20180701T0000/ng487-20180701T0000.nc3.nc',...
                'rutgers/ng488-20180701T0000/ng488-20180701T0000.nc3.nc',...
                'rutgers/ng616-20180701T0000/ng616-20180701T0000.nc3.nc',...
                'rutgers/ng617-20180701T0000/ng617-20180701T0000.nc3.nc',...
                'rutgers/ng618-20180701T0000/ng618-20180701T0000.nc3.nc',...
                });
%'rutgers/ng619-20180701T0000/ng619-20180701T0000.nc3.nc',...           
%}

%{
% SAB
lon_lim = [-81 -70];
lat_lim = [25 35];
id_list = char({'secoora/bass-20180808T0000/bass-20180808T0000.nc3.nc',...
                'secoora/pelagia-20180910T0000/pelagia-20180910T0000.nc3.nc',...
                'drudnick/sp022-20180912T1553/p022-20180912T1553.nc3.nc',...
                });
%}


                %{
% MAB
lon_lim = [-81 -70];
lat_lim = [30 42];
id_list = char({'rutgers/blue-20180806T1400/blue-20180806T1400.nc3.nc',...
                'rutgers/ru28-20180920T1334/ru28-20180920T1334.nc3.nc',...
                'rutgers/ru30-20180705T1825/ru30-20180705T1825.nc3.nc',...
                'rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc',...
                'rutgers/sylvia-20180802T0930/sylvia-20180802T0930.nc3.nc',...
                'rutgers/cp_336-20180724T1433/cp_336-20180724T1433.nc3.nc',...
                'rutgers/cp_376-20180724T1552/cp_376-20180724T1552.nc3.nc',...
                'rutgers/cp_389-20180724T1620/cp_389-20180724T1620.nc3.nc',...
                'secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc',...
                'secoora/ramses-20180704T0000/ramses-20180704T0000.nc3.nc',...
                'drudnick/sp010-20180620T1455/sp010-20180620T1455.nc3.nc',...
                'drudnick/sp022-20180422T1229/sp022-20180422T1229.nc3.nc',...
                'drudnick/sp066-20180629T1411/sp066-20180629T1411.nc3.nc',...
                'drudnick/sp069-20180411T1516/sp069-20180411T1516.nc3.nc',...
                });
%}
       
%%       
for l=1%:size(id_list,1)
    disp(l)
    clear temp_gridded salt_gridded
    gdata = [url,strtrim(id_list(l,:))];     

%% Glider Extract

   time = double(ncread(gdata,'time'));
   time = datenum(1970,01,01,0,0,time);
   
   % Finding subset of data for time period of interest
   tti = datenum(date_ini);
   tte = datenum(date_end);
   ok_time_glider = find(time >= tti & time < tte);
   
   if ~isempty(ok_time_glider) 

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');

temperature = double(ncread(gdata,'temperature'));
salinity = double(ncread(gdata,'salinity'));
pressure = double(ncread(gdata,'pressure'));

latitude = double(ncread(gdata,'latitude'));
longitude = double(ncread(gdata,'longitude'));

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
depth = ncread(catalog31,'depth');
time31 = ncread(catalog31,'time'); % hours since 2000-01-01 00:00:00

time_matlab31 = time31/24 + datenum(2000,01,01,0,0,0);

% Conversion from glider longitude and latitude to GOFS convention
if Glon < 0 
    target_lon = 360 + Glon;
else
    target_lon = Glon;
end
target_lat = Glat;

oklon31=round(interp1(lon31,1:length(lon31),target_lon));
oklat31=round(interp1(lat31,1:length(lat31),target_lat));
ok31 = find(time_matlab31 >= tti & time_matlab31 < tte);

target_temp31 = squeeze(double(ncread(catalog31,'water_temp',[oklon31 oklat31 1 ok31(1)],[1 1 inf length(ok31)])));
target_salt31 = squeeze(double(ncread(catalog31,'salinity',[oklon31 oklat31 1 ok31(1)],[1 1 inf length(ok31)])));

%% GOFS 3.0

lat30 = ncread(catalog30,'lat');
lon30 = ncread(catalog30,'lon');
depth30 = ncread(catalog30,'depth');
time30 = ncread(catalog30,'time'); % hours since 2000-01-01 00:00:00

time_matlab30 = time30/24 + datenum(2000,01,01,0,0,0);

ok30 = find(time_matlab30 >= tti & time_matlab30 < tte);
oklon30=round(interp1(lon30,1:length(lon30),target_lon));
oklat30=round(interp1(lat30,1:length(lat30),target_lat));

target_temp30 = squeeze(double(ncread(catalog30,'water_temp',[oklon30 oklat30 1 ok30(1)],[1 1 inf length(ok30)])));
target_salt30 = squeeze(double(ncread(catalog30,'salinity',[oklon30 oklat30 1 ok30(1)],[1 1 inf length(ok30)])));

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
freq_GOFS31 = (time_matlab31(2)-time_matlab31(1))*24;  %3 hourly

% GOFS3.0
freq_GOFS30 = (time_matlab30(2)-time_matlab30(1))*24; %daily

% Copernicus
%freq_Coper = (timecop(2)-timecop(1))*24; %daily

%% Mean profiles

depth_2d = repmat(depth,[1,length(ok31)]);
depth30_2d = repmat(depth30,[1,length(ok30)]);
%depthcop_2d = repmat(depthcop,[1,length(okcop)]);
%depthRTOFS1_2d = repmat(depthRTOFS1,[1,length(okRTOFS1)]);

temp31_mean = mean(target_temp31,2);
temp30_mean = mean(target_temp30,2);
%tempcop_mean = mean(target_tempcop,2);
%tempRTOFS1_mean = mean(target_tempRTOFS1,2);

salt31_mean = mean(target_salt31,2);
salt30_mean = mean(target_salt30,2);
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
    %saltu = saltok(oku,i);
    ok = isfinite(presu);
    %ok = isfinite(tempu);
    temp_gridded(:,i) = interp1(presu(ok),tempu(ok),pres_gridded);
    %salt_gridded(:,i) = interp1(presu(ok),saltu(ok),pres_gridded);
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

figure
set(gcf,'position',[139 144 1157 811])

ok26 = find(tempgl_mean >= 26.0 & tempgl_mean < 26.5); 

%%
subplot(121)
plot(temperature(:,ok_time_glider),-pressure(:,ok_time_glider),'.-g','markersize',mar_siz)
hold on
plot(target_temp31,-depth_2d,'.-c','markersize',mar_siz)
plot(target_temp30,-depth30_2d,'.-r','markersize',mar_siz)
%plot(target_tempcop,-depthcop_2d,'.-','color',[1 0.5 0],'markersize',mar_siz)
%plot(target_tempRTOFS1,-depthRTOFS1_2d,'.-','color',[1 0.5 1],'markersize',mar_siz)
h1 = plot(tempgl_mean,-pres_gridded,'.-k','markersize',mar_siz,'linewidth',4);
h2 = plot(temp31_mean,-depth,'.-b','markersize',mar_siz,'linewidth',4);
h3 = plot(temp30_mean,-depth30,'.-m','markersize',mar_siz,'linewidth',4);
%h4 = plot(tempcop_mean,-depthcop,'.-','color',[1 0.5 0],'markersize',mar_siz,'linewidth',4);
%h5 = plot(tempRTOFS1_mean,-depthRTOFS1,'.-','color',[1 0 0.2],'markersize',mar_siz,'linewidth',4);
if ~isempty(ok26)
plot(26.0,-pres_gridded(ok26(1)),'^r','markersize',10,'markerfacecolor','r')
dd1 = -max(pres_gridded):0;
tt1 = repmat(26.0,1,length(dd1));
plot(tt1,dd1,'--k')
tt2 = 15:26;
dd2 = repmat(-pres_gridded(ok26(1)),1,length(tt2));
plot(tt2,dd2,'--k')
end

set(gca,'fontsize',siz)
%lgd = legend([h1 h2 h3 h4],['cp376',' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
%lgd = legend([h1 h2 h3 h4],[inst_name,' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
%    ['HYCOM GOFS 3.1 Expt 93.0 (hindcast) ' datestr(time_matlab31(ok31(1)))],...
%    ['HYCOM GOFS 3.0 Expt 91.2 (hindcast) ' datestr(time_matlab30(ok30(1)))],...
%    'Location','SouthEast');
lgd = legend([h1 h2 h3],[inst_name,' ',datestr(time(ok_time_glider(1))),'-',datestr(time(ok_time_glider(end)))],...
    ['HYCOM GOFS 3.1 Expt 93.0 (hindcast) ' datestr(time_matlab31(ok31(1)))],...
    ['HYCOM GOFS 3.0 Expt 91.2 (hindcast) ' datestr(time_matlab30(ok30(1)))],...
    'Location','SouthEast');
%    ['Copernicus' datestr(timecop(okcop(1)))],...
%    'Location','SouthEast');
set(lgd,'fontsize',14)
title({'HYCOM GOFS 3.1 and 3.0 vs ';inst_name},'fontsize',siz)
%title({'HYCOM GOFS 3.1 and 3.0 and Copernicus vs ';inst_name},'fontsize',siz)
%title({'HYCOM GOFS 3.1 and 3.0 and Copernicus vs ';['cp376',' ',plat_type]},'fontsize',siz)
xlabel('Temperature (^oC)','fontsize',siz)
ylabel('Depth (m)','fontsize',siz);
grid on;
if max(max(pressure(:,ok_time_glider))) > 200
    ylim([-(200+200*0.2) 0])
else
    ylim([-max(max(pressure(:,ok_time_glider)))-max(max(pressure(:,ok_time_glider)))*0.2 0])
end
%xticks([15 20 25 26 30])

%%
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
title([inst_name,' ',' Track'],'fontsize',siz)
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
plot(salinity(:,ok_time_glider),-pressure(:,ok_time_glider),'.-g','markersize',mar_siz)
hold on
plot(target_salt31,-depth_2d,'.-c','markersize',mar_siz)
plot(target_salt30,-depth30_2d,'.-r','markersize',mar_siz)
%plot(target_saltcop,-depthcop_2d,'.-','color',[1 0.5 0],'markersize',mar_siz)
%plot(target_tempRTOFS1,-depthRTOFS1_2d,'.-','color',[1 0.5 1],'markersize',mar_siz)
h1 = plot(saltgl_mean,-pres_gridded,'.-k','markersize',mar_siz,'linewidth',4);
h2 = plot(salt31_mean,-depth,'.-b','markersize',mar_siz,'linewidth',4);
h3 = plot(salt30_mean,-depth30,'.-m','markersize',mar_siz,'linewidth',4);
%h4 = plot(saltcop_mean,-depthcop,'.-','color',[1 0.5 0],'markersize',mar_siz,'linewidth',4);
%h5 = plot(tempRTOFS1_mean,-depthRTOFS1,'.-','color',[1 0 0.2],'markersize',mar_siz,'linewidth',4);
set(gca,'fontsize',siz)
%lgd = legend([h1 h2 h3 h4],['cp376',' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
%lgd = legend([h1 h2 h3 h4],[inst_name,' ',plat_type,' ',datestr(time(ok_time_glider(1)))],...
%    ['HYCOM GOFS 3.1 Expt 93.0 (hindcast) ' datestr(time_matlab31(ok31(1)))],...
%    ['HYCOM GOFS 3.0 Expt 91.2 (hindcast) ' datestr(time_matlab30(ok30(1)))],...
%    ['Copernicus' datestr(timecop(okcop(1)))],...
%    'Location','SouthEast');
lgd = legend([h1 h2 h3],[inst_name,' ',datestr(time(ok_time_glider(1))),'-',datestr(time(ok_time_glider(end)))],...
    ['HYCOM GOFS 3.1 Expt 93.0 (hindcast) ' datestr(time_matlab31(ok31(1)))],...
    ['HYCOM GOFS 3.0 Expt 91.2 (hindcast) ' datestr(time_matlab30(ok30(1)))],...
    'Location','SouthEast');
set(lgd,'fontsize',14)
title({'HYCOM GOFS 3.1 and 3.0 vs ';inst_name},'fontsize',siz)
%title({'HYCOM GOFS 3.1 and 3.0 and Copernicus vs ';['cp376',' ',plat_type]},'fontsize',siz)
xlabel('Salinity (psu)','fontsize',siz)
ylabel('Depth (m)','fontsize',siz);
grid on;
if max(max(pressure(:,ok_time_glider))) > 200
    ylim([-(200+200*0.2) 0])
else
    ylim([-max(max(pressure(:,ok_time_glider)))-max(max(pressure(:,ok_time_glider)))*0.2 0])
end

%%
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
title([inst_name,' Track'],'fontsize',siz)
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

print([Fig_name,'_salt.png'],'-dpng','-r300') 
%print([Fig_name,'.eps'],'-depsc','-r300') 

   end
end

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
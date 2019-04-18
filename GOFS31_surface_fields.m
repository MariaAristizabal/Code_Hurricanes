
%% User input

date_ini = '10-Oct-2018 00:00:00';
date_end = '11-Oct-2018 00:00:00';

lat_min = 34;
lat_max = 38;
lon_min = -77;
lon_max = -74;

% Bathymetry data
%bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

% GOFS3.1 output model location
catalog31_ts = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

catalog31_uv = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/uv3z';

%% GOFS 3.1

latt31 = ncread(catalog31_ts,'lat');
lonn31 = ncread(catalog31_ts,'lon');
tim31 = ncread(catalog31_ts,'time'); % hours since 2000-01-01 00:00:00
time31 = tim31/24 + datenum(2000,01,01,0,0,0);

% Conversion from GOFS3.1 lat, lon convention to glider convection
long31(1:length(lonn31)) = nan;
for i=1:length(lonn31)
    if lonn31(i) > 180 
       long31(i) = -360 + lonn31(i);
    else
       long31(i) = lonn31(i);
    end
end
lat31 = latt31;
[lon31,ok] = sort(long31);

tti = datenum(date_ini);
tte = datenum(date_end);
oktime31 = find(time31 >= tti & time31 < tte);

oklat31 = find(lat31 >= lat_min & lat31 <= lat_max);
oklon31 = find(lon31 >= lon_min & lon31 <= lon_max);

tempp31 = squeeze(double(ncread(catalog31_ts,'water_temp',[1 1 1 oktime31(1)],[length(lon31) length(lat31) 1 1])));
saltt31 = squeeze(double(ncread(catalog31_ts,'salinity',[1 1 1 oktime31(1)],[length(lon31) length(lat31) 1 1])));

uu31 = squeeze(double(ncread(catalog31_uv,'water_u',[1 1 1 oktime31(1)],[length(lon31) length(lat31) 1 1])));
vv31 = squeeze(double(ncread(catalog31_uv,'water_v',[1 1 1 oktime31(1)],[length(lon31) length(lat31) 1 1])));

temp31 = tempp31(ok,:);
salt31 = saltt31(ok,:);
u31 = uu31(ok,:);
v31 = vv31(ok,:);
%temp31 = squeeze(double(ncread(catalog31,'water_temp',[oklon31(1) oklat31(1) 1 oktime31(1)],[oklon31(end) oklat31(end) 1 oktime31(end)])));
%salt31 = squeeze(double(ncread(catalog31,'salinity',[oklon31(1) oklat31(1) 1 oktime31(1)],[oklon31(end) oklat31(end) 1 oktime31(end)])));

%% Bathymetry data

%disp(bath_data);
bath_lat = ncread(bath_data,'lat');
bath_lon = ncread(bath_data,'lon');
bath_elev = ncread(bath_data,'elevation');

oklatbath = bath_lat >= lat_lim(1) & bath_lat <= lat_lim(2);
oklonbath = bath_lon >= lon_lim(1) & bath_lon <= lon_lim(2);

%% Tentative Michael path

lonMc = [-84.9 -85.2 -85.3 -85.9 -86.2 -86.4 -86.5 -86.5 -86.3 -86.2 ...
                  -86.0 -85.8 -85.5 -85.2 -84.9 -84.5 -84.1];
latMc = [21.2 22.2 23.2 24.1 25.0 26.0 27.1 28.3 28.8 29.1 29.4 29.6 ...
                  30.0 30.6 31.1 31.5 31.9];
tMc = ['2018/10/08/15/00';'2018/10/08/21/00';'2018/10/09/03/00';'2018/10/09/09/00';...
       '2018/10/09/15/00';'2018/10/09/21/00';'2018/10/10/03/00';'2018/10/10/09/00';...
       '2018/10/10/11/00';'2018/10/10/13/00';'2018/10/10/15/00';'2018/10/10/16/00';...
       '2018/10/10/18/00';'2018/10/10/20/00';'2018/10/10/22/00';'2018/10/11/00/00';...
       '2018/10/11/02/00'];
   
ttMc(1:length(tMc)) = nan;   
for i=1:length(tMc)
    ttMc(i) = datenum(tMc(i,:),'yyyy/mm/dd/HH/MM');
end

   
%datenum('2018/10/08/15/00','yyyy/mm/dd/HH/MM')  

%Convert time to UTC
%pst = pytz.timezone('America/New_York') # time zone
%utc = pytz.UTC 

%timeMc = [None]*len(tMc) 
%for x in range(len(tMc)):
%    timeMc[x] = datetime.datetime.strptime(tMc[x], '%Y/%m/%d/%H/%M') # time in time zone


figure
plot(lonMc,latMc,'o-','color',[0.5 0.5 0.5],'markerfacecolor',[0.5 0.5 0.5])
text(lonMc,latMc,datestr(ttMc,'dd, HH:MM'))

%% Temperature

figure 
contour(bath_lon,bath_lat,bath_elev',[0 0],'k','linewidth',2)
hold on
contourf(lon31,lat31,temp31')
colormap(jet)
%plot(lon31(oklon31),lat31(oklat31),'*k','markersize',10)
colorbar

%Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Temperature/GOFS31_lat_lon';
%print([Fig_name,'.png'],'-dpng','-r300')

%% Figure temperature detail
figure 
contour(bath_lon,bath_lat,bath_elev',[0 0],'k','linewidth',2)
hold on
pcolor(lon31,lat31,temp31')
shading interp
colormap(jet)
colorbar
caxis([15 30])
axis equal
xlim([-100 -40])
ylim([-5 45])
set(gca,'fontsize',16)
title(['Surface Temperature from GOFS 3.1 on ',datestr(time31(oktime31(1)))],'fontsize',16)

Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_temp_detail';
print([Fig_name,'.png'],'-dpng','-r300')

%% Figure temperature Gulf Mexico

figure 
contour(bath_lon,bath_lat,bath_elev',[0 0],'k','linewidth',2)
hold on
pcolor(lon31,lat31,temp31')
shading interp
colormap(jet)
colorbar
caxis([25 30])
axis equal
xlim([-100 -80])
ylim([17 32])
set(gca,'fontsize',16)
title(['Surface Temperature from GOFS 3.1 on ',datestr(time31(oktime31(1)))],'fontsize',16)
plot(lonMc,latMc,'o-','color','k','markerfacecolor','k')
text(lonMc(1:2:end-2)+0.2,latMc(1:2:end-2),datestr(ttMc(1:2:end-2),'dd, HH:MM'),'fontsize',12)

Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_temp_Gulf_Mexico';
print([Fig_name,'.png'],'-dpng','-r300')


%% Salinity

figure 
pcolor(lon31,lat31,salt31')
hold on
shading interp
colormap(jet)
colorbar
caxis([32 36])

%Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Salinity/GOFS31_salinity';
%print([Fig_name,'.png'],'-dpng','-r300')

%% Salinity caribbean

figure 
contour(bath_lon,bath_lat,bath_elev',[0 0],'k','linewidth',2)
hold on
pcolor(lon31,lat31,salt31')
shading interp
colormap(jet)
colorbar
caxis([32 37])
axis equal
xlim([-100 -40])
ylim([-5 45])
set(gca,'fontsize',16)
title(['Surface Salinity from GOFS 3.1 on ',datestr(time31(oktime31(1)))],'fontsize',16)

%Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_salinity_detail';
%print([Fig_name,'.png'],'-dpng','-r300')

%% Salinity Gulf of Mexico

figure 
contour(bath_lon,bath_lat,bath_elev',[0 0],'k','linewidth',2)
hold on
pcolor(lon31,lat31,salt31')
shading interp
colormap(jet)
colorbar
caxis([33 37])
axis equal
xlim([-100 -80])
ylim([17 32])
set(gca,'fontsize',16)
title(['Surface Salinity from GOFS 3.1 on ',datestr(time31(oktime31(1)))],'fontsize',16)
plot(lonMc,latMc,'o-','color','k','markerfacecolor','k')
text(lonMc(1:2:end-2)+0.2,latMc(1:2:end-2),datestr(ttMc(1:2:end-2),'dd, HH:MM'),'fontsize',12)

Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_salinity_Gulf_Mexico3';
print([Fig_name,'.png'],'-dpng','-r300')

%% u Velocity caribbean

figure 
contour(bath_lon,bath_lat,bath_elev',[0 0],'k','linewidth',2)
hold on
pcolor(lon31,lat31,u31')
shading interp
colormap(jet)
colorbar
%caxis([34 37])
axis equal
xlim([260-360 360-360])
ylim([-5 40])
title(['Surface Velocity from GOFS 3.1 on ',datestr(time31(oktime31(1)))],'fontsize',20)

%% v Velocity caribbean
dec = 5;
size = 5;

figure 
contour(bath_lon,bath_lat,bath_elev',[0 0],'k','linewidth',2)
hold on
pcolor(lon31,lat31,v31')
shading interp
quiver(lon31(1:dec:end),lat31(1:dec:end),u31(1:dec:end,1:dec:end)',v31(1:dec:end,1:dec:end)',size,'color','k','linewidth',2)
shading interp
axis equal
xlim([-70 -62])
ylim([16 22])
colormap(redblue)
c = colorbar;
ylabel(c,'m/s','fontsize',18)
caxis([-1.5 1.5])
title({'Surface Northward Velocity from GOFS 3.1 on ',datestr(time31(oktime31(1)))},'fontsize',20);

set(gca,'fontsize',16)

Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_vel_mag_detail2';
print([Fig_name,'.png'],'-dpng','-r300')

%% V vel

dec = 20;
size = 3;
figure 
contour(bath_lon,bath_lat,bath_elev',[0 0],'k','linewidth',2)
hold on
pcolor(lon31,lat31,v31')
shading interp
quiver(lon31(1:dec:end),lat31(1:dec:end),u31(1:dec:end,1:dec:end)',v31(1:dec:end,1:dec:end)',size,'color','k','linewidth',2)
axis equal
xlim([-100 0])
ylim([-5 40])
colormap(redblue)
c = colorbar;
ylabel(c,'m/s','fontsize',18)
caxis([-1.5 1.5])
title({'Surface Northward Velocity from GOFS 3.1 on ',datestr(time31(oktime31(1)))},'fontsize',20);

set(gca,'fontsize',16)

Fig_name = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/GOFS31_vel_mag_detail';
print([Fig_name,'.png'],'-dpng','-r300')

%% Velocity magnitude

dec = 20;
size = 3;
mag_vel = sqrt(u31.^2 + v31.^2);
figure 
contour(bath_lon,bath_lat,bath_elev',[0 0],'k','linewidth',2)
hold on
pcolor(lon31,lat31,mag_vel')
shading interp
quiver(lon31(1:dec:end),lat31(1:dec:end),u31(1:dec:end,1:dec:end)',v31(1:dec:end,1:dec:end)',size,'color','k','linewidth',2)
axis equal
xlim([-100 0])
ylim([-5 40])
%set(gcf,'position',[413         318        1401         667])
colormap(jet)
c = colorbar;
ylabel(c,'m/s','fontsize',18)
caxis([0 1.5])
title({'Surface Velocity Magnitude from GOFS 3.1 on ',datestr(time31(oktime31(1)))},'fontsize',20);

set(gca,'fontsize',16)

%% Velocity magnitude

dec = 5;
size = 3;
mag_vel = sqrt(u31.^2 + v31.^2);
figure 
%contour(bath_lon,bath_lat,bath_elev',[0 0],'k','linewidth',2)
hold on
pcolor(lon31,lat31,mag_vel')
shading interp
quiver(lon31(1:dec:end),lat31(1:dec:end),u31(1:dec:end,1:dec:end)',v31(1:dec:end,1:dec:end)',size,'color','k','linewidth',2)
axis equal
xlim([-70 -62])
ylim([16 22])
%set(gcf,'position',[413         318        1401         667])
colormap(jet)
c = colorbar;
ylabel(c,'m/s','fontsize',18)
caxis([0 1.5])
title({'Surface Velocity Magnitude from GOFS 3.1 on ',datestr(time31(oktime31(1)))},'fontsize',20);

set(gca,'fontsize',16)

%% Glider track as a function of time

clear all;

%% User input

% Glider data location

% Golf of Mexico
lon_lim = [-98 -78];
lat_lim = [18 32];
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

% RAMSES (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc';

% RU33 (MAB + SAB)
%lon_lim = [-81 -70];
%lat_lim = [30 42];
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';

% ng300 (Virgin Islands)
%lon_lim = [-68 -64];
%lat_lim = [15 20];
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng300-20180701T0000/ng300-20180701T0000.nc3.nc';

% Initial and final date
date_ini = '09-Oct-2018 00:00:00';
date_end = '11-Oct-2018 00:00:00';

% Bathymetry data
%bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

% Folder where to save figure
%folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Temperature/';
folder = '/Volumes/aristizabal-1/public_html/MARACOOS_proj/Figures/Model_glider_comp/Michael/';

%% Glider Extract

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

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
saltg = salinity(:,ok_time_glider);
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

%% Mean profiles

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
    ok = isfinite(tempu);
    temp_gridded(:,i) = interp1(presu(ok),tempu(ok),pres_gridded);
    salt_gridded(:,i) = interp1(presu(ok),saltu(ok),pres_gridded);
end

tempgl_mean = nanmean(temp_gridded,2);
saltgl_mean = nanmean(salt_gridded,2);


%% Bathymetry data

%disp(bath_data);
bath_lat = ncread(bath_data,'lat');
bath_lon = ncread(bath_data,'lon');
bath_elev = ncread(bath_data,'elevation');

%% Find up and downcast

saltg_up = nan(size(saltg,1),size(saltg,2));
saltg_down = nan(size(saltg,1),size(saltg,2));
saltg_up_interp = nan(size(saltg,1),size(saltg,2));
saltg_down_interp = nan(size(saltg,1),size(saltg,2));
for i=1:length(timeg)
    pos = find(presg(:,i) == max(presg(:,i)));
    ok_pres = isfinite(presg(:,i));
    ok_salt = isfinite(saltg(:,i));
    if pos(1)==1
        saltg_up(:,i) = saltg(:,i);
        %saltg_up_interp(1:size(presg(ok_pres,i)),i) = interp1(presg(ok_salt,i),saltg(ok_salt,i),presg(ok_pres,i));        
    else 
        saltg_down(:,i) = saltg(:,i);
        %saltg_down_interp(1:size(presg(ok_pres,i)),i) = interp1(presg(ok_salt,i),saltg(ok_salt,i),presg(ok_pres,i));      
    end
end


%% Figure bathymetry 

%{
figure
contourf(bath_lon,bath_lat,bath_elev')
hold on
contour(bath_lon,bath_lat,bath_elev',[0,-50,-100,-200,-1000,-2000,-4000,-8000],'color','k')
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
title([inst_name,' Track'],'fontsize',siz)
%title(['cp376',' ',plat_type,' Track'],'fontsize',siz)
%text(-77.5,42,['Glider Position  {\color{red}{\ast}}',{num2str(Glon)},{num2str(Glat)}],'fontsize',16)
%text(-67.5,20.5,['Glider Position  {\color{red}{\ast}}',{num2str(Glon)},{num2str(Glat)}],'fontsize',16)
text(lon_lim(1)+(lon_lim(end)-lon_lim(1))*0.05,lat_lim(end)-(lat_lim(end)-lat_lim(1))*0.1,...
    ['Glider Position  {\color{red}{\ast}}',{num2str(Glon)},{num2str(Glat)}],...
    'fontsize',16,'backgroundcolor','w')
%legend(pos,['Glider Position',{[num2str(Glon)]},{[num2str(Glat)]}],'location','northwest')

% Figure name
Fig_name = [folder,'glider_track',inst_name,'_',datestr(time(1),'mm-dd'),'_',datestr(time(end),'mm-dd')];

print([Fig_name,'.png'],'-dpng','-r300') 
%print([Fig_name,'.eps'],'-depsc','-r300') 
%}

%% Figure vertical profiles temperature

% Instrument name:
inst = strsplit(inst_id,'-');
inst_name = inst{1};

%plat = strsplit(plat_type,' ');

siz = 20;
mar_siz = 30;

figure
set(gcf,'position',[139 144 1157 811])

ok26 = find(tempgl_mean >= 26.0 & tempgl_mean < 26.1); 

subplot(121)
plot(temperature(:,ok_time_glider),-pressure(:,ok_time_glider),'.-g','markersize',mar_siz)
hold on
h1 = plot(tempgl_mean,-pres_gridded,'.-k','markersize',mar_siz,'linewidth',4);
plot(26.0,-pres_gridded(ok26(1)),'^r','markersize',10,'markerfacecolor','r')
dd1 = -max(pres_gridded):0;
tt1 = repmat(26.0,1,length(dd1));
plot(tt1,dd1,'--k')
tt2 = 15:26;
dd2 = repmat(-pres_gridded(ok26(1)),1,length(tt2));
plot(tt2,dd2,'--k')
set(gca,'fontsize',siz)
lgd = legend([h1],[inst_name,' ',plat_type,' ',datestr(time(ok_time_glider(end)))],...
    'Location','SouthEast');
set(lgd,'fontsize',14)
title({'Temperature ';[inst_name]},'fontsize',siz)
xlabel('Temperature (^oC)','fontsize',siz)
ylabel('Depth (m)','fontsize',siz);
grid on;
if max(max(pressure(:,ok_time_glider))) > 200
    ylim([-(200+200*0.2) 0])
else
    ylim([-max(max(pressure(:,ok_time_glider)))-max(max(pressure(:,ok_time_glider)))*0.2 0])
end
xticks([15 20 25 26 30])
text(16,-10,[datestr(timeg(1)),'-',datestr(timeg(end))],'fontsize',16)

subplot(122)
%plot(salinity(:,ok_time_glider),-pressure(:,ok_time_glider),'.-g','markersize',mar_siz)
h1 = plot(saltg_up(:,1),-presg(:,1),'.-r','markersize',mar_siz,'linewidth',2);
hold on
h2 = plot(saltg_down(:,1),-presg(:,1),'.-b','markersize',mar_siz,'linewidth',2);
h3 = plot(saltgl_mean,-pres_gridded,'.-k','markersize',mar_siz,'linewidth',2);
lgd = legend([h1 h2 h3],{'Upcast','Downcast','Average profile'},'Location','SouthEast');
plot(saltg_up,-presg,'.-r','markersize',mar_siz,'linewidth',2);
hold on
plot(saltg_down,-presg,'.-b','markersize',mar_siz,'linewidth',2);
plot(saltgl_mean,-pres_gridded,'.-k','markersize',mar_siz,'linewidth',2);
set(gca,'fontsize',siz)
lgd = legend([h1 h2 h3],{'Upcast','Downcast','Average profile'},'Location','SouthEast');
set(lgd,'fontsize',14)
title({'Salinity ';[inst_name]},'fontsize',siz)
xlabel('Salinity (psu)','fontsize',siz)
ylabel('Depth (m)','fontsize',siz);
grid on;
if max(max(pressure(:,ok_time_glider))) > 200
    ylim([-(200+200*0.2) 0])
else
    ylim([-max(max(pressure(:,ok_time_glider)))-max(max(pressure(:,ok_time_glider)))*0.2 0])
end

% Figure name
Fig_name = [folder,'vertical_profiles_',inst_name,'_',datestr(timeg(1),'mm-dd'),'_',datestr(timeg(end),'mm-dd')];

%print([Fig_name,'_.png'],'-dpng','-r300') 
%print([Fig_name,'.eps'],'-depsc','-r300') 

%% Upcast ans downcast for a couple of profiles

figure
set(gcf,'position',[560   173   674   775])
h1 = plot(saltg_up(:,1),-presg(:,1),'.-r','markersize',mar_siz,'linewidth',2);
hold on
h3 = plot(saltg_up(:,44),-presg(:,44),'.-g','markersize',mar_siz,'linewidth',2);
h2 = plot(saltg_down(:,2),-presg(:,2),'.-b','markersize',mar_siz,'linewidth',2);
h4 = plot(saltg_down(:,43),-presg(:,43),'.-k','markersize',mar_siz,'linewidth',2);
set(gca,'fontsize',siz)
lgd = legend([h1 h2 h3 h4],{['Upcast ',datestr(datestr(timeg(1)))],...
                            ['Downcast ',datestr(datestr(timeg(2)))],...
                            ['Upcast ',datestr(datestr(timeg(44)))],...
                            ['Downcast ',datestr(datestr(timeg(43)))]},...
                            'Location','west');
set(lgd,'fontsize',14)
title({'Salinity ';inst_name},'fontsize',siz)
xlabel('Salinity (psu)','fontsize',siz)
ylabel('Depth (m)','fontsize',siz);
grid on;
if max(max(pressure(:,ok_time_glider))) > 200
    ylim([-(200+200*0.2) 0])
else
    ylim([-max(max(pressure(:,ok_time_glider)))-max(max(pressure(:,ok_time_glider)))*0.2 0])
end

% Figure name
Fig_name = [folder,'up_downcast_salinity_profile_',inst_name,'_',datestr(timeg(1),'mm-dd'),'_',datestr(timeg(end),'mm-dd')];

%print([Fig_name,'_.png'],'-dpng','-r300') 


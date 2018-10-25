%% Glider track as a function of time

clear all;

%% User input

% Glider data location
% Golf of Mexico
lon_lim = [-98 -78];
lat_lim = [18 32];
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

% Initial and final date
date_ini = '09-Oct-2018 00:00:00';
date_end = '11-Oct-2018 00:00:00';

delta_t = 24; %hours

% Folder where to save figure
%folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/Temperature/';
folder = '/Volumes/aristizabal/public_html/MARACOOS_proj/Figures/Model_glider_comp/Michael/';

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
temp_gridded(length(pres_gridded),size(presg,2)) = nan;
salt_gridded(length(pres_gridded),size(presg,2)) = nan;

for i=1:size(pressure(:,ok_time_glider),2)
    [presu,oku] = unique(presg(:,i));
    tempu = tempg(oku,i);
    saltu = saltg(oku,i);
    %ok = isfinite(tempu);
    ok = isfinite(presu);
    temp_gridded(:,i) = interp1(presu(ok),tempu(ok),pres_gridded);
    salt_gridded(:,i) = interp1(presu(ok),saltu(ok),pres_gridded);
end

%% dt averages

dt = datenum(0,0,0,delta_t,0,0); % dt hours
nwindows = round((timeg(end) -timeg(1))*24/delta_t);

temp_dt_mean(size(temp_gridded,1),nwindows) = nan;
salt_dt_mean(size(salt_gridded,1),nwindows) = nan;
for n=1:nwindows
    twindow = timeg >= datenum(date_ini) & timeg <= datenum(date_ini) + n*dt;
    temp_dt_mean(:,n) = nanmean(temp_gridded(:,twindow),2);
    salt_dt_mean(:,n) = nanmean(salt_gridded(:,twindow),2);
end

%%

pres_matrix = repmat(pres_gridded,size(temp_gridded,2),1);
twindow1 = timeg >= datenum(date_ini) & timeg <= timeg(69);
twindow2 = timeg > timeg(69) & timeg <= timeg(end);

figure
plot(temp_gridded(:,twindow1),-pres_matrix(twindow1,:)','.-','color','b')
hold on
plot(temp_gridded(:,twindow2),-pres_matrix(twindow2,:)','.-','color','r')
grid on

%%
figure
for t=1:size(temp_gridded,2)
plot(temp_gridded(:,t),-pres_matrix(t,:)','.-')
title(['t = ',num2str(t),' ',datestr(timeg(t))],'fontsize',20)
hold on
pause
end
%% Figure vertical profiles temperature

% Instrument name:
inst = strsplit(inst_id,'-');
inst_name = inst{1};

%plat = strsplit(plat_type,' ');

siz = 20;
mar_siz = 10;
color = jet(64);

figure
set(gcf,'position',[139 144 1157 811])

%ok26 = find(tempgl_mean >= 26.0 & tempgl_mean < 26.1); 

%subplot(121)
%for n=1:nwindows
%    pos = floor(size(color,1)/nwindows)*n;
%    plot(temp_dt_mean(:,n),-pres_gridded,'.-','markersize',mar_siz,'linewidth',2,'color',color(pos,:));
%hold on
%end


pres_matrix = repmat(pres_gridded,size(temp_gridded,2),1);
twindow1 = timeg >= datenum(date_ini) & timeg <= timeg(69);
twindow2 = timeg > timeg(69) & timeg <= timeg(end);

subplot(211)
plot(temp_gridded(:,twindow1),-pres_matrix(twindow1,:)','.','color','b','markersize',15)
hold on
plot(temp_gridded(:,twindow2),-pres_matrix(twindow2,:)','.','color','r','markersize',15)
dd1 = -max(pres_gridded):0;
tt1 = repmat(26.0,1,length(dd1));
plot(tt1,dd1,'--k','linewidth',2)
h1 = plot(temp_gridded(:,twindow1(1)),-pres_matrix(twindow1(1),:)','.-','color','b','markersize',10);
h2 = plot(temp_gridded(:,twindow2(80)),-pres_matrix(twindow2(80),:)','.-','color','r','markersize',10);
lgd = legend([h1 h2],{['Ahead-of-eye-center ',datestr(timeg(1),'mm/dd HH:MM '),' - ',datestr(timeg(69),'mm/dd HH:MM ')],...
                      ['After-eye-center ',datestr(timeg(70),'mm/dd HH:MM '),' - ',datestr(timeg(end),'mm/dd HH:MM ')]},...
    'Location','northwest');
grid on
set(gca,'fontsize',siz)
title(inst_name,'fontsize',siz)
xlabel('Temperature (^oC)','fontsize',siz)
ylabel('Depth (m)','fontsize',siz);
grid on;
if max(max(pressure(:,ok_time_glider))) > 200
    ylim([-(200+200*0.2) 0])
else
    ylim([-max(max(pressure(:,ok_time_glider)))-max(max(pressure(:,ok_time_glider)))*0.2 0])
end
xticks([15:2:25,26:2:30])
set(lgd,'FontSize',20) 

%for n=1:nwindows
%    pos = floor(size(color,1)/nwindows)*n;
%    plot(salt_dt_mean(:,n),-pres_gridded,'.-','markersize',mar_siz,'linewidth',2,'color',color(pos,:));
%hold on
%end

subplot(212)
plot(salt_gridded(:,twindow1),-pres_matrix(twindow1,:)','.','color','b','markersize',15)
hold on
plot(salt_gridded(:,twindow2),-pres_matrix(twindow2,:)','.','color','r','markersize',15)
grid on
set(gca,'fontsize',siz)
xlabel('Salinity (psu)','fontsize',siz)
ylabel('Depth (m)','fontsize',siz);
grid on;
if max(max(pressure(:,ok_time_glider))) > 200
    ylim([-(200+200*0.2) 0])
else
    ylim([-max(max(pressure(:,ok_time_glider)))-max(max(pressure(:,ok_time_glider)))*0.2 0])
end
xticks([36:0.2:37])

% Figure name
Fig_name = [folder,'vertical_profiles_',inst_name,'_',datestr(timeg(1),'mm-dd'),'_',datestr(timeg(end),'mm-dd')];
%print([Fig_name,'_.png'],'-dpng','-r300') 
%print([Fig_name,'.eps'],'-depsc','-r300') 

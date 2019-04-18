%% Glider/Model Data Comparison

clear all;
 %% User input

% Initial and final date

% Florence
date_ini = '06-Sep-2018 00:00:00';
date_end = '15-Sep-2018 00:00:00';

% Michael
%date_ini = '05-Oct-2018 00:00:00';
%date_end = '13-Oct-2018 00:00:00';

%date_ini = ''; %if empty, date_ini is the firts time stamp in data
%date_end = ''; %if empty, date_end is the last time stamp in data

% GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

% GOFS3.1 output model location
catalog30 = 'http://tds.hycom.org/thredds/dodsC/GLBu0.08/expt_91.2/ts3z';

% Bathymetry data
%bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

% Copernicus
%copern = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/global-analysis-forecast-phy-001-024_1537209157740.nc';
copern = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/global-analysis-forecast-phy-001-024_1538489276306.nc';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/'; %Temperature/';
%folder = '/Volumes/aristizabal/public_html/MARACOOS_proj/Figures/Model_glider_comp/';

% Glider data location
url = 'https://data.ioos.us/thredds/dodsC/deployments/';

% Directories where RTOFS files reside 
Dir= '/Volumes/aristizabal/GOFS/';

% Gulf of Mexico
%lon_lim = [-98,-78];
%lat_lim = [18,32];

%Initial and final date
%dateini = '2018/10/10/00/00';
%dateend = '2018/10/17/00/00';

url = 'https://data.ioos.us/thredds/dodsC/deployments/';

%id_list = 'rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';
%id_list = 'rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';
%id_list = 'rutgers/blue-20180806T1400/blue-20180806T1400.nc3.nc';
%id_list = 'secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc';
%id_list = 'rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc';
id_list = 'rutgers/ng302-20180701T0000/ng302-20180701T0000.nc3.nc';

% Increment file
%ncoda_file = [Dir,'GLBy0.08_seatmp_2018101012_ncoda_inc_930_ALL.nc'];  %Michael
ncoda_file = [Dir,'GLBy0.08_seatmp_2018091112_ncoda_inc_930_ALL.nc']; %Florence

%% Glider Extract

gdata = strtrim([url,id_list(1,:)]);
inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

temperature = double(ncread(gdata,'temperature'));
pressure = double(ncread(gdata,'pressure'));
time = double(ncread(gdata,'time'));
time = datenum(1970,01,01,0,0,time);
latitude = double(ncread(gdata,'latitude'));
longitude = double(ncread(gdata,'longitude'));

% Finding subset of data for time period of interest
if isempty(date_ini)
   tti = time(1);
else
   tti = datenum(date_ini);
end

if isempty(date_end)
   tte = time(end);
else
   tte = datenum(date_end);
end

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

%% GOFS 3.1

%ncdisp(catalog31);

lat31 = ncread(catalog31,'lat');
lon31 = ncread(catalog31,'lon');
depth31 = ncread(catalog31,'depth');
tim31 = ncread(catalog31,'time'); % hours since 2000-01-01 00:00:00

time31 = tim31/24 + datenum(2000,01,01,0,0,0);

%oktime31 = find(time31 >= time(1) & time31 < time(end));
oktime31 = find(time31 >= tti & time31 < tte);

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

target_temp31(length(depth31),length(oktime31))=nan;
for i=1:length(oklon31)
    disp(length(oklon31))
    disp(['GOFS3.1 ',num2str(i)])
    if isfinite(oklon31(i)) && isfinite(oklat31(i)) && isfinite(oktime31(i))
       target_temp31(:,i) = squeeze(double(ncread(catalog31,'water_temp',[oklon31(i) oklat31(i) 1 oktime31(i)],[1 1 inf 1])));
    else
       target_temp31(:,i) = nan;
    end
end


%% Reading ncoda increments file

lat_ncoda = ncread(ncoda_file,'Latitude');
lon_ncoda = ncread(ncoda_file,'Longitude');
depth_ncoda = ncread(ncoda_file,'Depth');
tim = ncread(ncoda_file,'MT'); % hours since 2000-01-01 00:00:00
time_ncoda = tim + datenum(1900,12,31,0,0,0);
temp_incr = ncread(ncoda_file,'pot_temp');

%% Interpolating location of glider on model grid
%oktime = timeg >= datenum(2018,10,10,09,00,00) & timeg <=%datenum(2018,10,10,12,00,00); %Michael
oktime = timeg >= datenum(2018,09,11,09,00,00) & timeg <=datenum(2018,09,11,12,00,00);  %Florence
xlon = mean(long(oktime));
ylat = mean(latg(oktime));

xpos = round((interp1(lon_ncoda-360,1:length(lon_ncoda),xlon)));
ypos = round((interp1(lat_ncoda,1:length(lat_ncoda),ylat)));


%% Mean profiles
%{
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

saltok = salinity(:,ok_time_glider);
salt_gridded(length(pres_gridded),size(presok,2)) = nan;
%}
pres_gridded = 0:0.5:max(max(pressure(:,ok_time_glider)));
presok = pressure(:,ok_time_glider);
tempok = temperature(:,ok_time_glider);
temp_gridded(length(pres_gridded),size(presok,2)) = nan;

for i=1:size(pressure(:,ok_time_glider),2)
    [presu,oku] = unique(presok(:,i));
    tempu = tempok(oku,i);
    %saltu = saltok(oku,i);
    %k = isfinite(presu);
    ok = isfinite(tempu);
    temp_gridded(:,i) = interp1(presu(ok),tempu(ok),pres_gridded);
    %salt_gridded(:,i) = interp1(presu(ok),saltu(ok),pres_gridded);
end

%% Obtain delta Temp between 09Z and 12Z for GOFS 3.1 on Oct 10, 2018 at all depths
%{
ok09 = time31(oktime31) == datenum(2018,10,10,09,00,00);
ok12 = time31(oktime31) == datenum(2018,10,10,12,00,00);

delta_temp = target_temp31(:,ok12) - target_temp31(:,ok09);

figure
%plot(delta_temp,-depth31,'o-')
hold on
plot(squeeze(temp_incr(xpos,ypos,:)),-depth_ncoda,'o-')
%legend('Temp(12Z) - Temp(09Z) from GOFS 3.1','NCODA Temperature increments')
ylim([-500 0])
%}

%% 
figure
set(gcf,'position',[680 179 586 799])
plot(squeeze(temp_incr(xpos,ypos,:)),-depth_ncoda,'o-','linewidth',4,'markersize',8)
set(gca,'fontsize',20)
ylabel('Depth (m)')
xlabel('Temperature Increments (^oC)')
title(['Temperature Increments on ',datestr(time_ncoda),' at ',inst_name])
ylim([-300 0])
Fig_name = [folder,'Temp_incr_profile_',inst_name];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%% Michael
dep = 10;
pos31 = interp1(depth31,1:length(depth31),dep);
posglider = interp1(pres_gridded,1:length(pres_gridded),dep);

figure
set(gcf,'position',[680         630        1059         348])
yvec = 23:0.1:30.5;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
y = [23 23 30.5 30.5];
x = [datenum(2018,10,05,09,0,0) datenum(2018,10,05,12,0,0) datenum(2018,10,05,12,0,0) datenum(2018,10,05,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
hold on
x = [datenum(2018,10,06,09,0,0) datenum(2018,10,06,12,0,0) datenum(2018,10,06,12,0,0) datenum(2018,10,06,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,07,09,0,0) datenum(2018,10,07,12,0,0) datenum(2018,10,07,12,0,0) datenum(2018,10,07,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,08,09,0,0) datenum(2018,10,08,12,0,0) datenum(2018,10,08,12,0,0) datenum(2018,10,08,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,09,09,0,0) datenum(2018,10,09,12,0,0) datenum(2018,10,09,12,0,0) datenum(2018,10,09,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,10,09,0,0) datenum(2018,10,10,12,0,0) datenum(2018,10,10,12,0,0) datenum(2018,10,10,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,11,09,0,0) datenum(2018,10,11,12,0,0) datenum(2018,10,11,12,0,0) datenum(2018,10,11,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,12,09,0,0) datenum(2018,10,12,12,0,0) datenum(2018,10,12,12,0,0) datenum(2018,10,12,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
yvec = 23.0:0.1:30.5;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
%plot(xvec,yvec,'.-k','linewidth',4)
hold on
xvec = repmat(datenum(2018,10,10,12,0,0),1,length(yvec));
%plot(xvec,yvec,'--k','linewidth',4)
xvec = repmat(datenum(2018,10,09,12,0,0),1,length(yvec));
%plot(xvec,yvec,'--k','linewidth',4)
h1 = plot(timeg,temp_gridded(posglider,:),'^-','linewidth',2,'markersize',4);
h2 = plot(time31(oktime31),target_temp31(pos31,:),'^-','linewidth',2,'markersize',4);
datetick
set(gca,'fontsize',18)
legend([h1 h2], 'ng288','GOFS 3.1')
title(['Depth = ',num2str(dep)' m'],'fontsize',20)
ylim([23 30.5])
xlim([datenum(2018,10,05) datenum(2018,10,13)])

ylabel('Sea Water Temperature (^oC)')

Fig_name = [folder,'Temp_time_series_GOFS31_',inst_name,'_',num2str(dep),' m'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%% Florence 10 m
dep = 10;
pos31 = interp1(depth31,1:length(depth31),dep);
posglider = interp1(pres_gridded,1:length(pres_gridded),dep);

figure
set(gcf,'position',[680         630        1059         348])
%yvec = 23:0.1:30.5;
%xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
%plot(xvec,yvec,'.-k','linewidth',4)
ymin1 = round(min(temp_gridded(posglider,:)))-1;
ymin2 = round(min(target_temp31(pos31,:)))-1;
ymin = min([ymin1,ymin2]);
ymax1 = round(max(temp_gridded(posglider,:)))+1;
ymax2 = round(max(target_temp31(pos31,:)))+1;
ymax = max([ymax1,ymax2]);
y = [ymin ymin ymax ymax];
x = [datenum(2018,09,06,09,0,0) datenum(2018,09,06,12,0,0) datenum(2018,09,06,12,0,0) datenum(2018,09,06,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
hold on
x = [datenum(2018,09,07,09,0,0) datenum(2018,09,07,12,0,0) datenum(2018,09,07,12,0,0) datenum(2018,09,07,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,08,09,0,0) datenum(2018,09,08,12,0,0) datenum(2018,09,08,12,0,0) datenum(2018,09,08,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,09,09,0,0) datenum(2018,09,09,12,0,0) datenum(2018,09,09,12,0,0) datenum(2018,09,09,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,10,09,0,0) datenum(2018,09,10,12,0,0) datenum(2018,09,10,12,0,0) datenum(2018,09,10,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,11,09,0,0) datenum(2018,09,11,12,0,0) datenum(2018,09,11,12,0,0) datenum(2018,09,11,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,12,09,0,0) datenum(2018,09,12,12,0,0) datenum(2018,09,12,12,0,0) datenum(2018,09,12,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,13,09,0,0) datenum(2018,09,13,12,0,0) datenum(2018,09,13,12,0,0) datenum(2018,09,13,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,14,09,0,0) datenum(2018,09,14,12,0,0) datenum(2018,09,14,12,0,0) datenum(2018,09,14,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
%yvec = 23.0:0.1:30.5;
%xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
%plot(xvec,yvec,'.-k','linewidth',4)
hold on
%xvec = repmat(datenum(2018,10,10,12,0,0),1,length(yvec));
%plot(xvec,yvec,'--k','linewidth',4)
%xvec = repmat(datenum(2018,10,09,12,0,0),1,length(yvec));
%plot(xvec,yvec,'--k','linewidth',4)
h1 = plot(timeg,temp_gridded(posglider,:),'^-','linewidth',2,'markersize',4);
h2 = plot(time31(oktime31),target_temp31(pos31,:),'^-','linewidth',2,'markersize',4);
datetick
set(gca,'fontsize',18)
legend([h1 h2], inst_name,'GOFS 3.1')
title(['Depth = ',num2str(dep),' m'],'fontsize',20)
ylim([ymin ymax])
%xlim([datenum(2018,10,05) datenum(2018,10,13)])

ylabel('Sea Water Temperature (^oC)')

Fig_name = [folder,'Temp_time_series_GOFS31_',inst_name,'_',num2str(dep),' m'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%% Florence
dep = 80;
pos31 = interp1(depth31,1:length(depth31),dep);
posglider = interp1(pres_gridded,1:length(pres_gridded),dep);

figure
set(gcf,'position',[680         630        1059         348])
%yvec = 23:0.1:30.5;
%xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
%plot(xvec,yvec,'.-k','linewidth',4)
ymin1 = round(min(temp_gridded(posglider,:)))-1;
ymin2 = round(min(target_temp31(pos31,:)))-1;
ymin = min([ymin1,ymin2]);
ymax1 = round(max(temp_gridded(posglider,:)))+1;
ymax2 = round(max(target_temp31(pos31,:)))+1;
ymax = max([ymax1,ymax2]);
y = [ymin ymin ymax ymax];
x = [datenum(2018,09,06,09,0,0) datenum(2018,09,06,12,0,0) datenum(2018,09,06,12,0,0) datenum(2018,09,06,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
hold on
x = [datenum(2018,09,07,09,0,0) datenum(2018,09,07,12,0,0) datenum(2018,09,07,12,0,0) datenum(2018,09,07,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,08,09,0,0) datenum(2018,09,08,12,0,0) datenum(2018,09,08,12,0,0) datenum(2018,09,08,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,09,09,0,0) datenum(2018,09,09,12,0,0) datenum(2018,09,09,12,0,0) datenum(2018,09,09,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,10,09,0,0) datenum(2018,09,10,12,0,0) datenum(2018,09,10,12,0,0) datenum(2018,09,10,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,11,09,0,0) datenum(2018,09,11,12,0,0) datenum(2018,09,11,12,0,0) datenum(2018,09,11,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,12,09,0,0) datenum(2018,09,12,12,0,0) datenum(2018,09,12,12,0,0) datenum(2018,09,12,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,13,09,0,0) datenum(2018,09,13,12,0,0) datenum(2018,09,13,12,0,0) datenum(2018,09,13,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,09,14,09,0,0) datenum(2018,09,14,12,0,0) datenum(2018,09,14,12,0,0) datenum(2018,09,14,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
%yvec = 23.0:0.1:30.5;
%xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
%plot(xvec,yvec,'.-k','linewidth',4)
hold on
%xvec = repmat(datenum(2018,10,10,12,0,0),1,length(yvec));
%plot(xvec,yvec,'--k','linewidth',4)
%xvec = repmat(datenum(2018,10,09,12,0,0),1,length(yvec));
%plot(xvec,yvec,'--k','linewidth',4)
h1 = plot(timeg,temp_gridded(posglider,:),'^-','linewidth',2,'markersize',4);
h2 = plot(time31(oktime31),target_temp31(pos31,:),'^-','linewidth',2,'markersize',4);
datetick
set(gca,'fontsize',18)
legend([h1 h2], inst_name,'GOFS 3.1','location','northwest')
title(['Depth = ',num2str(dep),' m'],'fontsize',20)
ylim([ymin ymax])
%xlim([datenum(2018,10,05) datenum(2018,10,13)])

ylabel('Sea Water Temperature (^oC)')

Fig_name = [folder,'Temp_time_series_GOFS31_',inst_name,'_',num2str(dep),'_m'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%%
dep = 30;
pos31 = interp1(depth31,1:length(depth31),dep);
posglider = interp1(pres_gridded,1:length(pres_gridded),dep);

figure
set(gcf,'position',[680         630        1059         348])
plot(timeg,temp_gridded(posglider,:),'^-','linewidth',2,'markersize',4)
hold on
plot(time31(oktime31),target_temp31(pos31,:),'^-','linewidth',2,'markersize',4)
datetick
legend('ng288','GOFS 3.1')
title(['Depth = ',num2str(dep),' m'],'fontsize',20)
ylim([23 30.5])

%%
dep = 50;
pos31 = interp1(depth31,1:length(depth31),dep);
posglider = interp1(pres_gridded,1:length(pres_gridded),dep);

figure
set(gcf,'position',[680         630        1059         348])
plot(timeg,temp_gridded(posglider,:),'^-','linewidth',2,'markersize',4)
hold on
plot(time31(oktime31),target_temp31(pos31,:),'^-','linewidth',2,'markersize',4)
datetick
legend('ng288','GOFS 3.1')
title(['Depth = ',num2str(dep),' m'],'fontsize',20)
ylim([23 30.5])

%%
dep = 100;
pos31 = interp1(depth31,1:length(depth31),dep);
posglider = interp1(pres_gridded,1:length(pres_gridded),dep);

figure
set(gcf,'position',[680         630        1059         348])
yvec = 23:0.1:30.5;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
y = [23 23 30.5 30.5];
x = [datenum(2018,10,05,09,0,0) datenum(2018,10,05,12,0,0) datenum(2018,10,05,12,0,0) datenum(2018,10,05,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
hold on
x = [datenum(2018,10,06,09,0,0) datenum(2018,10,06,12,0,0) datenum(2018,10,06,12,0,0) datenum(2018,10,06,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,07,09,0,0) datenum(2018,10,07,12,0,0) datenum(2018,10,07,12,0,0) datenum(2018,10,07,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,08,09,0,0) datenum(2018,10,08,12,0,0) datenum(2018,10,08,12,0,0) datenum(2018,10,08,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,09,09,0,0) datenum(2018,10,09,12,0,0) datenum(2018,10,09,12,0,0) datenum(2018,10,09,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,10,09,0,0) datenum(2018,10,10,12,0,0) datenum(2018,10,10,12,0,0) datenum(2018,10,10,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,11,09,0,0) datenum(2018,10,11,12,0,0) datenum(2018,10,11,12,0,0) datenum(2018,10,11,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
x = [datenum(2018,10,12,09,0,0) datenum(2018,10,12,12,0,0) datenum(2018,10,12,12,0,0) datenum(2018,10,12,09,0,0)];
patch(x,y,[0.9 0.9 0.9])
yvec = 23.0:0.1:30.5;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
%plot(xvec,yvec,'.-k','linewidth',4)
hold on
xvec = repmat(datenum(2018,10,10,12,0,0),1,length(yvec));
%plot(xvec,yvec,'--k','linewidth',4)
xvec = repmat(datenum(2018,10,09,12,0,0),1,length(yvec));
%plot(xvec,yvec,'--k','linewidth',4)
h1 = plot(timeg,temp_gridded(posglider,:),'^-','linewidth',2,'markersize',4);
h2 = plot(time31(oktime31),target_temp31(pos31,:),'^-','linewidth',2,'markersize',4);
datetick
set(gca,'fontsize',18)
legend([h1 h2], 'ng288','GOFS 3.1')
title(['Depth = ',num2str(dep),' m'],'fontsize',20)
ylim([23 30.5])
xlim([datenum(2018,10,05) datenum(2018,10,13)])

ylabel('Sea Water Temperature (^oC)')

Fig_name = [folder,'Temp_time_series_ng288_GOFS31_100m'];
wysiwyg
%print([Fig_name,'.png'],'-dpng','-r300') 

%%%%%%%%%%%%%%

%% Temperature time series of temp at 20 m

marker.MarkerSize = 16;
siz_text = 20;
siz_title =16;

okg20 = find(pres_gridded == 20.0);
okG20 = find(depth31 == 20.0);

figure
set(gcf,'position',[607 282 1227 703])
yvec = 27.5:0.1:29.5;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
hold on
xvec = repmat(datenum(2018,10,10,12,0,0),1,length(yvec));
plot(xvec,yvec,'--k','linewidth',4)
xvec = repmat(datenum(2018,10,09,12,0,0),1,length(yvec));
plot(xvec,yvec,'--k','linewidth',4)
h1 = plot(timeg(oktt),temp_full(okg20,:),'^-b','linewidth',4,'markersize',8,'markerfacecolor','b');
h2 = plot(time31(oktime31),target_temp31_20m,'^-r','linewidth',4,'markersize',8,'markerfacecolor','r');
set(gca,'fontsize',siz_text)
legend([h1 h2],{['Glider ',inst_name],'GOFS 3.1'},'fontsize',siz_text)
xlim([datenum(2018,10,09) datenum(2018,10,11)])
tt_vec = datenum(2018,10,09,00,00,00):datenum(0,0,0,6,0,0):datenum(2018,10,11,0,0,0); 
xticks(tt_vec)
xticklabels(datestr(tt_vec,'dd-HH:MM'))
xl = xlabel('2018 October Date-Time (DD-HH:MM UTC)');

ylabel('Sea Water Temperature (^oC)')
title('Time Series of Water Temperature at 20 Meters Depth','fontsize',30) 
ylim([27.5 29.5])
yticks(27.5:0.5:30)
grid on
ax = gca;
ax.GridAlpha = 0.4 ;
ax.GridLineStyle = '--';


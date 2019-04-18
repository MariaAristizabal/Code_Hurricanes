%% Glider track as a function of time

clear all;

%% User input

% Glider data location

% ng467 (Virgin Islands)
lon_lim = [-68 -64];
lat_lim = [15 20];
gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc';
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng487-20180701T0000/ng487-20180701T0000.nc3.nc';
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng300-20180701T0000/ng300-20180701T0000.nc3.nc';
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng291-20180701T0000/ng291-20180701T0000.nc3.nc';

%date_ini = '11-Sep-2018 00:00:00';
%date_end = '12-Sep-2018 00:00:00';
date_ini = ''; %if empty, date_ini is the firts time stamp in data
date_end = ''; %if empty, date_end is the last time stamp in data

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

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';

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
u = double(ncread(gdata,'u'));
v = double(ncread(gdata,'v'));


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
saltg = salinity(:,ok_time_glider);
presg = pressure(:,ok_time_glider);
latg = latitude(ok_time_glider);
long = longitude(ok_time_glider);
timeg = time(ok_time_glider);
ug = u(ok_time_glider);
vg = v(ok_time_glider);


% Final lat and lon of deployment for time period of interest 
% This lat and lon is used to find the grid point in the model output
% we can consider changing it to the average lat, lon for the entire
% deployment
Glat = latitude(ok_time_glider(end));
Glon = longitude(ok_time_glider(end));

%% GOFS 3.1

ncdisp(catalog31_uv)

lat31 = ncread(catalog31_uv,'lat');
lon31 = ncread(catalog31_uv,'lon');
depth31 = ncread(catalog31_uv,'depth');
tim31 = ncread(catalog31_uv,'time'); % hours since 2000-01-01 00:00:00
time31 = tim31/24 + datenum(2000,01,01,0,0,0);
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
       target_temp31(:,i) = squeeze(double(ncread(catalog31_ts,'water_temp',[oklon31(i) oklat31(i) 1 oktime31(i)],[1 1 inf 1])));
    else
       target_temp31(:,i) = nan;
    end
end
%%
target_salt31(length(depth31),length(oktime31))=nan;
for i=1:length(oklon31)
    disp(length(oklon31))
    disp(['GOFS3.1 ',num2str(i)])
    if isfinite(oklon31(i)) && isfinite(oklat31(i)) && isfinite(oktime31(i))
       target_salt31(:,i) = squeeze(double(ncread(catalog31_ts,'salinity',[oklon31(i) oklat31(i) 1 oktime31(i)],[1 1 inf 1])));
    else
       target_salt31(:,i) = nan;
    end
end

%%

target_u31(length(depth31),length(oktime31))=nan;
target_v31(length(depth31),length(oktime31))=nan;
for i=1:length(oklon31)
    disp(length(oklon31))
    disp(['GOFS31',' i= ',num2str(i)])
    if isfinite(oklon31(i)) && isfinite(oklat31(i)) && isfinite(oktime31(i))
       target_u31(:,i) = squeeze(double(ncread(catalog31_uv,'water_u',[oklon31(i) oklat31(i) 1 oktime31(i)],[1 1 inf 1])));
       target_v31(:,i) = squeeze(double(ncread(catalog31_uv,'water_v',[oklon31(i) oklat31(i) 1 oktime31(i)],[1 1 inf 1])));
    else
       target_u31(:,i) = nan; 
       target_v31(:,i) = nan; 
    end
end

%% Rotate velocities

[ur,vr,thetad]=maria_rot_max_var(ug,vg);

mean_depth_u31 = nanmean(target_u31)';
mean_depth_v31 = nanmean(target_v31)';
%[ur31,vr31,thetad31]=maria_rot_max_var(mean_depth_u31,mean_depth_v31);

vr31 = -mean_depth_u31 * sin(thetad) + mean_depth_v31 *cos(thetad);
ur31 = u*cos(thetad)+v*sin(thetad);

%%
yvec = -0.4:0.1:0.4;
xvec = -0.4:0.1:0.4;

okd = depth31 <= 200;
mean_depth_u31 = nanmean(target_u31(okd,:))';
mean_depth_v31 = nanmean(target_v31(okd,:))';

figure
set(gcf,'position',[193 436 585 549])
plot(ug,vg,'.b','markersize',20)
hold on
plot(mean_depth_u31,mean_depth_v31,'.r','markersize',20)
plot(zeros(1,length(yvec)),yvec,'.-k','linewidth',1)
hold on
plot(xvec,zeros(1,length(xvec)),'.-k','linewidth',1)
xlim([-0.4 0.4]);
ylim([-0.4 0.4]);
xticks = (-0.4:0.2:0.4);
yticks = (-0.4:0.2:0.4);
%plot(ur,vr,'.b')
%legend('u and v','u and v rotated')
ylabel('v (m/s)')
xlabel('u (m/s)')
grid on
set(gca,'fontsize',16)
title('200 m Depth Average Velocity, 18 Jul-17 Sep')
legend(inst_name,'GOFS 3.1')

wysiwyg

Fig_name = [folder,'Depth_aver_200m_vel_GOFS31_',inst_name,'_',datestr(timeg(1),'mm-dd-yy'),'-',datestr(timeg(end),'mm-dd-yy')];
print([Fig_name,'.png'],'-dpng','-r300') 


%% 
figure
pcolor(time31(oktime31),-depth31,target_u31)
shading interp
datetick
c = colorbar;
caxis([-0.3 0.3])
set(c,'ytick',-0.3:0.1:0.3)
ylim([-500 0])
c.Label.String = 'Water u (m/s)';
colormap(redblue)

%% 
figure
pcolor(time31(oktime31),-depth31,target_v31)
shading interp
datetick
c = colorbar;
caxis([-0.3 0.3])
set(c,'ytick',-0.3:0.1:0.3)
%ylim([-500 0])
c.Label.String = 'Water v (m/s)';
colormap(redblue)

%%

%[vglid,ok]= unique(vg);
%timeglid = timeg(ok);

figure
plot(timeg,vg,'.-')
hold on
plot(time31(oktime31),mean_depth_v31,'.-')
datetick
legend('v glider','v GOFS31')

%%
figure
plot(timeg,ug,'.-')
hold on
plot(time31(oktime31),mean_depth_u31,'.-')
datetick
legend('u glider','u GOFS31')

%%
figure
%quiver(timeglid,xvec,uglid,uglid,'ShowArrowHead','off','linewidth',1.5,'AutoScale','off')
quiver(yvec,yvec,uglid',vglid','ShowArrowHead','off','linewidth',1.5)

%%
xvec31 = 1:length(time31(oktime31));
yvec31 = sqrt(mean_depth_u31.^2 + mean_depth_v31.^2);
xvec = 1:length(xvec31);
yvec = sqrt(ug.^2 + vg.^2);

figure
quiver(xvec31(1:10)',yvec31(1:10),mean_depth_u31(1:10)./yvec31(1:10),mean_depth_v31(1:10)./yvec31(1:10))
hold on
quiver(xvec(1:10)',yvec(1:10),ug(1:10)./yvec(1:10),vg(1:10)./yvec(1:10))

%%  Direction

okd = depth31 <= 200;
mean_depth_u31 = nanmean(target_u31(okd,:))';
mean_depth_v31 = nanmean(target_v31(okd,:))';

angleg = atand(vg./ug);
angle31 = atand(mean_depth_v31./mean_depth_u31);

figure
plot(timeg,angleg,'.-')
hold on
plot(time31(oktime31),angle31,'.-')
plot(time31(oktime31),zeros(length(time31(oktime31))),'-k','linewidth',2)
datetick
legend(inst_name,'GOFS 3.1')
ylabel('Angle (^o)')

set(gca,'fontsize',16)
title('Angle of 200 m Depth Average Velocity, 18 Jul-17 Sep')

wysiwyg

Fig_name = [folder,'Angle Depth_aver_200m_vel_GOFS31_',inst_name,'_',datestr(timeg(1),'mm-dd-yy'),'-',datestr(timeg(end),'mm-dd-yy')];
print([Fig_name,'.png'],'-dpng','-r300') 

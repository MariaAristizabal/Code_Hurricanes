
%% Ocean heat content (OHC)

clear all;

%% User input

% Glider data location

% Gulf of Mexico
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ng288-20180801T0000/ng288-20180801T0000.nc3.nc';

% RAMSES (MAB + SAB)
gdata = 'https://data.ioos.us/thredds/dodsC/deployments/secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc';

% RU33 (MAB + SAB)
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc';

% blue (MAB + SAB)
%gdata = 'https://data.ioos.us/thredds/dodsC/deployments/rutgers/blue-20180806T1400/blue-20180806T1400.nc3.nc';

% ng467 (Virgin Islands)
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc';

% ng302 (Virgin Islands)
%gdata = 'http://data.ioos.us/thredds/dodsC/deployments/rutgers/ng302-20180701T0000/ng302-20180701T0000.nc3.nc';

% Initial and final date
%date_ini = '05-Oct-2018 00:00:00';
%date_end = '13-Oct-2018 00:00:00';
date_ini = '06-Sep-2018 00:00:00';
close adate_end = '15-Sep-2018 00:00:00';
%date_ini = ''; %if empty, date_ini is the firts time stamp in data
%date_end = ''; %if empty, date_end is the last time stamp in data

% GOFS3.1 outout model location
catalog31 = 'http://tds.hycom.org/thredds/dodsC/GLBv0.08/expt_93.0/ts3z';

% Bathymetry data
%bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/GEBCO_2014_2D_-100.0_0.0_-60.0_45.0.nc';
bath_data = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/nc_files/GEBCO_2014_2D_-100.0_0.0_-10.0_70.0.nc';

% Folder where to save figure
folder = '/Users/aristizabal/Desktop/MARACOOS_project/Maria_scripts/Figures/Model_glider_comp/';
%folder = '/Volumes/aristizabal/public_html/MARACOOS_proj/Figures/Model_glider_comp/';

%% Glider Extract

inst_id = ncreadatt(gdata,'/','id');
plat_type = ncreadatt(gdata,'/','platform_type');
inst = strsplit(inst_id,'-');
inst_name = inst{1};

temperature = double(ncread(gdata,'temperature'));
salinity = double(ncread(gdata,'salinity'));
density = double(ncread(gdata,'density'));
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
saltg = salinity(:,ok_time_glider);
densg = density(:,ok_time_glider);
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
target_salt31(length(depth31),length(oktime31))=nan;

for i=1:length(oklon31) % 1:length(oklon31)
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

%% Ocean heat content glider

cp = 3985; %J/(kg K)

OHC1_gl(1:length(timeg)) = nan;
OHC2_gl(1:length(timeg)) = nan;
OHC3_gl(1:length(timeg)) = nan;
min_temp(1:length(timeg)) = nan;
for t=1:length(timeg)
     ok = isfinite(tempg(:,t));
     tempok = tempg(ok,t);
     presok = presg(ok,t);
     densok = densg(ok,t);
     
     [presu,oku] = unique(presok);
     tempu = tempok(oku);
     densu = densok(oku);
     
      presu = presu(isfinite(presu));
      tempu = tempu(isfinite(presu));
      densu = densu(isfinite(presu));
           
     if ~isempty(presu(isfinite(presu))) && length(presu(isfinite(presu))) > 2
        presu = presu(isfinite(presu));
        tempu = tempu(isfinite(presu));
        densu = densu(isfinite(presu));
        
        %calculate dz for integration
        delta_z = presu(2:end) - presu(1:end-1);
        
        presg_mid = presu(1:end-1) + delta_z/2; 
        tempg_mid = interp1(presu,tempu,presg_mid);
        densg_mid = interp1(presu,densu,presg_mid);
     else
        presg_mid = nan;
        tempg_mid = nan;
        densg_mid = nan;
     end
   
     ok26 = tempg_mid >= 26.0;
     if sum(ok26)  ~= 0
       min_temp(t) = min(tempg_mid(ok26));
    else
        min_temp(t) = nan;
    end
     rho0 = nanmean(densg_mid(ok26)+1000);
     OHC3_gl(t) = cp * rho0 * nansum((tempg_mid(ok26)-26) .* abs(delta_z(ok26)));
end

%% Ocean Heat Content GOFS 3.1 

depth31;
target_temp31;
target_salt31;

dens31(1:size(target_temp31,1),1:size(target_temp31,2)) = nan;
for i=1:length(time31(oktime31))
    dens31(:,i) = sw_dens(target_salt31(:,i),target_temp31(:,i),depth31);
end

% Calculate delta density and delta z to perform vertical integration
delta_rho31 = dens31(2:end,:) - dens31(1:end-1,:);
delta_z31 = depth31(2:end,:) - depth31(1:end-1,:);

% Calculate the position of mid point z, that is the vertical
% position where the integration is centered 
depth_mid31 = depth31(1:end-1,:) + delta_z31/2; 

temp_mid31(1:size(depth_mid31,1),1:length(time31(oktime31))) = nan;
dens_mid31(1:size(depth_mid31,1),1:length(time31(oktime31))) = nan;
for i=1:length(time31(oktime31))
    temp_mid31(:,i) = interp1(depth31,target_temp31(:,i),depth_mid31);
    dens_mid31(:,i) = interp1(depth31,dens31(:,i),depth_mid31);
end

OHC1_31(1:length(time31(oktime31))) = nan;
OHC2_31(1:length(time31(oktime31))) = nan;
OHC3_31(1:length(time31(oktime31))) = nan;
for i=1:length(time31(oktime31))
    ok26 = temp_mid31(:,i) >= 26.0; 
    rho0 = nanmean(dens_mid31(ok26,i));
    OHC1_31(i) = cp * nansum((temp_mid31(ok26,i)-26) .* (dens_mid31(ok26,i)) .* abs(delta_z31(ok26)));
    OHC2_31(i) = cp * nansum((temp_mid31(ok26,i)-26) .* (depth_mid31(ok26)) .* abs(delta_rho31(ok26,i)));
    OHC3_31(i) = cp * rho0 * nansum((temp_mid31(ok26,i)-26) .* abs(delta_z31(ok26)));
end 

OHC3_31(OHC1_31 ==0) = nan;

%% Criteria for OHC from profiles that are not complete

bad_OHC = min_temp > 26.5 | OHC3_gl < nanmean(OHC3_gl)-2*nanstd(OHC3_gl);
OHC3_gl(bad_OHC) = nan;
ok_OHC = isfinite(OHC3_gl);

%% Interpolate OHC3_gl to GOFS 3.1 time

%OHC1_gl_int = interp1(timeg(ok_OHC),OHC1_gl(ok_OHC),time31(oktime31));
if sum(ok_OHC) > 3
   OHC3_gl_int = interp1(timeg(ok_OHC),OHC3_gl(ok_OHC),time31(oktime31));
else
   OHC3_gl_int = nan;
end

%% Calculation of T100 glider

% find depth= 100 m
okg100 = presg < 100;

Tg100(1:length(timeg)) = nan;
for i=1:length(timeg)
    Tg100(i) = nanmean(tempg(okg100(:,i),i));
end

%% Criteria for Tg100 from profiles that are not complete
% Need to think more about this

bad_Tg100 = Tg100 < nanmean(Tg100)-2*nanstd(Tg100) | Tg100 > nanmean(Tg100)+2*nanstd(Tg100);
Tg100(bad_Tg100) = nan;
ok_Tg100 = isfinite(Tg100);

%% Calculation T100 GOFS 3.1

okm100 = depth31 < 100;

target_temp100 = nanmean(target_temp31(okm100,:));

%% Interpolate Tg100 to GOFS 3.1 time

Tg100_gl_int = interp1(timeg,Tg100,time31(oktime31)); 
%Tg100_gl_int = interp1(timeg(ok_OHC),Tg100(ok_OHC),time31(oktime31));

%%
%{
figure
plot(timeg,OHC1_gl,'.-b')
hold on
plot(timeg,OHC2_gl,'.-r')
plot(timeg,OHC3_gl,'.-g')
datetick

%%

figure
plot(time31(oktime31),OHC1_31,'.-b')
hold on
plot(time31(oktime31),OHC2_31,'.-r')
plot(time31(oktime31),OHC3_31,'.-g')
datetick

%%

figure
plot(timeg,OHC3_gl,'*c')
hold on
plot(time31(oktime31),OHC3_gl_int,'*-b')
plot(time31(oktime31),OHC3_31,'*-r')
datetick
%}
%%

figure
plot(timeg,OHC3_gl,'*c')
hold on
plot(time31(oktime31),OHC3_gl_int,'*-b')
plot(time31(oktime31),OHC3_31,'*-r')
datetick


%% Michael ng288 OHC

figure
set(gcf,'position',[680         630        1059         348])
yvec = (4:0.1:11)*10^8;
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
y = [4 4 11 11]*10^8;
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
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
plot(timeg(ok_OHC),OHC3_gl(ok_OHC),'oc','markerfacecolor','c')
h1 =  plot(time31(oktime31),OHC3_gl_int,'o-b','markerfacecolor','b','linewidth',2);
h2 = plot(time31(oktime31),OHC3_31,'o-r','markerfacecolor','r','linewidth',2);
datetick
set(gca,'fontsize',18)
legend([h1 h2], inst_name,'GOFS 3.1')
title('Ocean Heat Content','fontsize',20)
%ylim([23 30.5])
xlim([datenum(2018,10,05) datenum(2018,10,13)])

ylabel('Ocean Heat Content (J/m^2)')

Fig_name = [folder,'Ocean_heat_content_GOFS31_vs_',inst_name,'_',' m'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%% Florence OHC

yveci = min([OHC3_gl OHC3_31])-1*10^8;
yvece = max(OHC3_gl)+1*10^8;
dy = abs(OHC3_gl(2) - OHC3_gl(1));

figure
set(gcf,'position',[680         630        1059         348])
yvec = (yveci:dy:yvece);
xvec = repmat(datenum(2018,09,13,18,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
y = [yveci yveci yvece yvece];
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
%xvec = repmat(datenum(2018,09,11,6,0,0),1,length(yvec));
%plot(xvec,yvec,'.-k','linewidth',4)
plot(timeg(ok_OHC),OHC3_gl(ok_OHC),'oc','markerfacecolor','c')
h1 =  plot(time31(oktime31),OHC3_gl_int,'o-b','markerfacecolor','b','linewidth',2);
h2 = plot(time31(oktime31),OHC3_31,'o-r','markerfacecolor','r','linewidth',2);
datetick
set(gca,'fontsize',18)
legend([h1 h2], inst_name,'GOFS 3.1')
title('Ocean Heat Content','fontsize',20)
%ylim([23 30.5])
xlim([datenum(2018,09,06) datenum(2018,09,15)])

ylabel('Ocean Heat Content (J/m^2)')

Fig_name = [folder,'Ocean_heat_content_GOFS31_vs_',inst_name,'_',' m'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 

%% Michael ng288 T100

figure
set(gcf,'position',[680         630        1059         348])
yvec = (26:0.1:30);
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
y = [26 26 30 30];
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
xvec = repmat(datenum(2018,10,10,6,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
plot(timeg(ok_OHC),Tg100(ok_OHC),'oc','markerfacecolor','c')
h1 =  plot(time31(oktime31),Tg100_gl_int,'o-b','markerfacecolor','b','linewidth',2);
h2 = plot(time31(oktime31),target_temp100,'o-r','markerfacecolor','r','linewidth',2);
datetick
set(gca,'fontsize',18)
legend([h1 h2], 'ng288','GOFS 3.1')
title('Depth Average Temperature over Top 100 meters','fontsize',20)
%ylim([23 30.5])
xlim([datenum(2018,10,05) datenum(2018,10,13)])

ylabel('Temperature (^oC)')

Fig_name = [folder,'T100_GOFS31_vs_',inst_name,'_',' m'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 


%% Florence ng288 T100

yveci = min([Tg100 target_temp100])-1;
yvece = max([Tg100 target_temp100])+1;
dy = 0.1;

figure
set(gcf,'position',[680         630        1059         348])
yvec = (yveci:dy:yvece);
xvec = repmat(datenum(2018,09,13,18,0,0),1,length(yvec));
plot(xvec,yvec,'.-k','linewidth',4)
y = [yveci yveci yvece yvece];
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
plot(timeg,Tg100,'oc','markerfacecolor','c')
h1 =  plot(time31(oktime31),Tg100_gl_int,'o-b','markerfacecolor','b','linewidth',2);
h2 = plot(time31(oktime31),target_temp100,'o-r','markerfacecolor','r','linewidth',2);
datetick
set(gca,'fontsize',18)
legend([h1 h2], inst_name,'GOFS 3.1')
title('Depth Average Temperature over Top 100 meters','fontsize',20)
%ylim([23 30.5])
%xlim([datenum(2018,10,05) datenum(2018,10,13)])

ylabel('Temperature (^oC)')

Fig_name = [folder,'T100_GOFS31_vs_',inst_name,'_',' m'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 
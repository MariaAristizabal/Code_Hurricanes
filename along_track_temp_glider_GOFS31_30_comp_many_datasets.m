%% Glider/Model Data Comparison

clear all;
 %% User input

% Initial and final date
%date_ini = '01-Aug-2018 00:00:00';
%date_end = '09-Oct-2018 00:00:00';
date_ini = ''; %if empty, date_ini is the firts time stamp in data
date_end = ''; %if empty, date_end is the last time stamp in data

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
           'rutgers/ng512-20180701T0000/ng512-20180701T0000.nc3.nc',...
           });
%}

%'gcoos_dmac/Reveille-20180627T1500/Reveille-20180627T1500.nc3/nc',...
%'rutgers/ng489-20180701T0000/ng489-20180701T0000.nc3.nc',...
%'rutgers/ng596-20180701T0000/ng596-20180701T0000.nc3.nc',...


%{
%Caribbean:
lon_lim = [-68 -64];
lat_lim = [15 20];
id_list = char({'aoml/SG630-20180716T1220/SG630-20180716T1220.nc3.nc',...
                'rutgers/ng300-20180701T0000/ng300-20180701T0000.nc3.nc',...
                'aoml/SG610-20180719T1146/SG610-20180719T1146.nc3.nc',...
                'aoml/SG635-20180716T1248/SG635-20180716T1248.nc3.nc',...
                'aoml/SG649-20180731T1418/SG649-20180731T1418.nc3.nc',...
                'rutgers/ng291-20180701T0000/ng291-20180701T0000.nc3.nc',...
                'rutgers/ng302-20180701T0000/ng302-20180701T0000.nc3.nc',...
                'rutgers/ng467-20180701T0000/ng467-20180701T0000.nc3.nc',...
                'rutgers/ng487-20180701T0000/ng487-20180701T0000.nc3.nc',...
                'rutgers/ng488-20180701T0000/ng488-20180701T0000.nc3.nc',...
                'rutgers/ng616-20180701T0000/ng616-20180701T0000.nc3.nc',...
                'rutgers/ng617-20180701T0000/ng617-20180701T0000.nc3.nc',...
                'rutgers/ng618-20180701T0000/ng618-20180701T0000.nc3.nc',...
                'rutgers/ng619-20180701T0000/ng619-20180701T0000.nc3.nc',...
                });
%}

%{
%SAB:
lon_lim = [-81 -70];
lat_lim = [25 35];
id_list = char({'secoora/bass-20180808T0000/bass-20180808T0000.nc3.nc',...
                'secoora/pelagia-20180910T0000/pelagia-20180910T0000.nc3.nc',...
                });
%}
            
%'drudnick/sp022-20180912T1553/p022-20180912T1553.nc3.nc',...


%{
% MAB
lon_lim = [-81 -70];
lat_lim = [30 42];
id_list = char({'rutgers/blue-20180806T1400/blue-20180806T1400.nc3.nc',...
                'rutgers/ru28-20180920T1334/ru28-20180920T1334.nc3.nc',...
                'rutgers/ru30-20180705T1825/ru30-20180705T1825.nc3.nc',...
                'rutgers/ru33-20180801T1323/ru33-20180801T1323.nc3.nc',...
                'rutgers/sylvia-20180802T0930/sylvia-20180802T0930.nc3.nc',...
                'rutgers/cp_376-20180724T1552/cp_376-20180724T1552.nc3.nc',...
                'rutgers/cp_389-20180724T1620/cp_389-20180724T1620.nc3.nc',...
                'rutgers/cp_336-20180724T1433/cp_336-20180724T1433.nc3.nc',...
                'secoora/ramses-20180907T0000/ramses-20180907T0000.nc3.nc',...
                'secoora/ramses-20180704T0000/ramses-20180704T0000.nc3.nc',...
                'drudnick/sp010-20180620T1455/sp010-20180620T1455.nc3.nc',...
                'drudnick/sp022-20180422T1229/sp022-20180422T1229.nc3.nc',...
                'drudnick/sp066-20180629T1411/sp066-20180629T1411.nc3.nc',...
                'drudnick/sp069-20180411T1516/sp069-20180411T1516.nc3.nc',...
                });
%}
            
%'drudnick/sp065-20180310T1828/sp065-20180310T1828.nc3.nc.'...

       
%%
for l=1:size(id_list,1)%:size(id_list,1)
    clear target_temp31 target_temp30 target_tempcop time_mat time_vec depth_vec temp_vec
    clear target_lon target_lat
    gdata = strtrim([url,id_list(l,:)]);
 
%% Glider Extract

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

%% GOFS 3.0

lat30 = ncread(catalog30,'lat');
lon30 = ncread(catalog30,'lon');
depth30 = ncread(catalog30,'depth');
time30 = ncread(catalog30,'time'); % hours since 2000-01-01 00:00:00
time30 = time30/24 + datenum(2000,01,01,0,0,0);

oktime30 = find(time30 >= tti & time30 < tte);

sublon30=interp1(timeg,target_lon,time30(oktime30));
sublat30=interp1(timeg,target_lat,time30(oktime30));

oklon30=round(interp1(lon30,1:length(lon30),sublon30));
oklat30=round(interp1(lat30,1:length(lat30),sublat30));

target_temp30(length(depth30),length(oktime30))=nan;
for i=1:length(oklon30)
    disp(length(oklon30))
    disp(['GOFS3.0 ',num2str(i)])
    if isfinite(oklon30(i)) && isfinite(oklat30(i)) && isfinite(oktime30(i))
       target_temp30(:,i) = squeeze(double(ncread(catalog30,'water_temp',[oklon30(i) oklat30(i) 1 oktime30(i)],[1 1 inf 1])));
    else
       target_temp30(:,i) = nan; 
    end
end

%% Copernicus

%ncdisp(copern);
%{
latcop = ncread(copern,'latitude');
loncop = ncread(copern,'longitude');
depthcop = ncread(copern,'depth');

timecop = ncread(copern,'time');
timecop = double(timecop)/24 + datenum('01-Jan-1950');

%oktimecop = find(timecop >= time(1) & timecop < time(end));
oktimecop = find(timecop >= (time(1)-datenum(0,0,0,3,0,0)) & timecop < time(end));

subloncop=interp1(time,longitude,timecop(oktimecop));
sublatcop=interp1(time,latitude,timecop(oktimecop));
%subloncop=interp1(time,longitude,[time(1) ;timecop(oktimecop)]);
%sublatcop=interp1(time,latitude,[time(1) ;timecop(oktimecop)]);

okloncop=round(interp1(loncop,1:length(loncop),subloncop));
oklatcop=round(interp1(latcop,1:length(latcop),sublatcop));

%target_tempcop(length(depthcop),length([time(1) ;timecop(oktimecop)]))=nan;
target_tempcop(length(depthcop),length(oktimecop))=nan;
for i=1:length(okloncop)
    %if i==1
    %   target_tempcop(:,i) = squeeze(double(ncread(copern,'thetao',[okloncop(i) oklatcop(i) 1 2],[1 1 inf 1])));
    %else
       %target_tempcop(:,i) = squeeze(double(ncread(copern,'thetao',[okloncop(i) oklatcop(i) 1 oktimecop(i-1)],[1 1 inf 1])));
       target_tempcop(:,i) = squeeze(double(ncread(copern,'thetao',[okloncop(i) oklatcop(i) 1 oktimecop(i)],[1 1 inf 1])));
    %end
end
%}
%% Bathymetry data

%disp(bath_data);
bath_lat = ncread(bath_data,'lat');
bath_lon = ncread(bath_data,'lon');
bath_elev = ncread(bath_data,'elevation');

oklatbath = bath_lat >= lat_lim(1) & bath_lat <= lat_lim(2);
oklonbath = bath_lon >= lon_lim(1) & bath_lon <= lon_lim(2);

%% Figure with 2 models: GOFS3.1, GOFS3.0 

siz_title = 20;
siz_text = 16;

color = colormap(jet(length(latg)+1));
norm = (timeg-timeg(1))./(timeg(end)-timeg(1));
pos = round(norm.*length(latg))+1;

time_mat = repmat(timeg,1,size(tempg,1))';
time_vec = reshape(time_mat,1,size(time_mat,1)*size(time_mat,2));

depth_vec = reshape(presg,1,size(presg,1)*size(presg,2));
tempg_vec = reshape(tempg,1,size(tempg,1)*size(tempg,2));
tempg_vec(tempg_vec <= 0) = nan;

marker.MarkerSize = 16;

figure
set(gcf,'position',[366 82 1134 902])

%%
subplot(411)
set(gca,'position', [0.38 0.78 0.19 0.19])
contour(bath_lon(oklonbath),bath_lat(oklatbath),bath_elev(oklonbath,oklatbath)',[0,-50,-100,-200,-1000,-2000,-4000,-8000],...
        'color','k','linewidth',2)
%contour(bath_lon,bath_lat,bath_elev',[0 0],'color','k','linewidth',2)
hold on
%contourf(bath_lon,bath_lat,bath_elev',[0 max(max(bath_elev))],'color',[0.5 0.2 0])
%contourf(bath_lon,bath_lat,bath_elev',[min(min(bath_elev)) -0.1])
axis equal
ylim(lat_lim)
xlim(lon_lim)
set(gca,'fontsize',siz_text)

%%
for i=1:length(latg)
    plot(long(i),latg(i),'.','markersize',12,'color',color(pos(i),:,:))
end
colormap('jet')
c = colorbar('v');
caxis([timeg(1) timeg(end)])
tt_vec = unique([timeg(1),timeg(1)+(timeg(end)-timeg(1))/5:(timeg(end)-timeg(1))/5:timeg(end),timeg(end)]);
time_lab = datestr(tt_vec,'mm/dd/yy');
set(c,'ytick',tt_vec)
datetick(c,'keepticks')
set(c,'yticklabel',time_lab)

%%
okt26 = tempg_vec > 25.9 & tempg_vec < 26.1;
temp26 = tempg_vec(okt26);
t26 = time_vec(okt26);
dep26 = depth_vec(okt26);

subplot(412)
[h,c_h] = fast_scatter(time_vec',-depth_vec',tempg_vec','colorbar','vert','marker',marker);
hold on
plot(t26,-dep26,'.k')

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title(['Along track temperature profile ',inst_name],'fontsize',siz_title)

c = colorbar;
colormap('jet')
caxis([floor(min(tempg_vec)) ceil(max(tempg_vec))])
cc_vec = unique(round(floor(min(tempg_vec)):(max(tempg_vec)-min(tempg_vec))/5:ceil(max(tempg_vec))));
set(c,'ytick',cc_vec)

tt_vec = unique(floor([time_vec(1),time_vec(1)+(time_vec(end)-time_vec(1))/10:(time_vec(end)-time_vec(1))/10:time_vec(end),time_vec(end)]));
xticks(tt_vec)
set(gca,'xticklabel',{[]})
xlim([tt_vec(1) time_vec(end)])

ylim([-max(depth_vec) 0])
yticks(floor(-max(depth_vec):max(depth_vec)/5:0))

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

%%
subplot(413)
pcolor(time31(oktime31),-depth31,target_temp31)
hold on
contour(time31(oktime31),-depth31,target_temp31,[26,26],'color','k','linewidth',3)
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title('GOFS 3.1','fontsize',siz_title)

c = colorbar;
colormap('jet')
caxis([floor(min(tempg_vec)) ceil(max(tempg_vec))])
cc_vec = unique(round(floor(min(tempg_vec)):(max(tempg_vec)-min(tempg_vec))/5:ceil(max(tempg_vec))));
set(c,'ytick',cc_vec)
c.Label.String = 'Water Temperature (^oC)';
c.Label.FontSize = siz_text;

tt_vec = unique(floor([time_vec(1),time_vec(1)+(time_vec(end)-time_vec(1))/10:(time_vec(end)-time_vec(1))/10:time_vec(end),time_vec(end)]));
xticks(tt_vec)
set(gca,'xticklabel',{[]})
xlim([tt_vec(1) time_vec(end)])

ylim([-max(depth_vec) 0])
yticks(floor(-max(depth_vec):max(depth_vec)/5:0))

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

%%
subplot(414)
pcolor(time30(oktime30),-depth30,target_temp30)
hold on
contour(time30(oktime30),-depth30,target_temp30,[26,26],'color','k','linewidth',3)
shading interp

set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title('GOFS 3.0','fontsize',siz_title)

c = colorbar;
colormap('jet')
caxis([floor(min(tempg_vec)) ceil(max(tempg_vec))])
cc_vec = unique(round(floor(min(tempg_vec)):(max(tempg_vec)-min(tempg_vec))/5:ceil(max(tempg_vec))));
set(c,'ytick',cc_vec)

tt_vec = unique(floor([time_vec(1),time_vec(1)+(time_vec(end)-time_vec(1))/10:(time_vec(end)-time_vec(1))/10:time_vec(end),time_vec(end)]));
xticks(tt_vec)
xticklabels(datestr(tt_vec,'mm/dd/yy'))
xlim([tt_vec(1) time_vec(end)])
xtickangle(45)

ylim([-max(depth_vec) 0])
yticks(floor(-max(depth_vec):max(depth_vec)/5:0))

set(gca,'position',[0.1300    0.1100    0.735    0.15])
set(c,'position',[0.872    0.1098    0.0141    0.15])

set(gca,'TickDir','out') 
set(gca,'xgrid','on','ygrid','on','layer','top')

ax = gca;
ax.GridAlpha = 0.3;

%%
% Figure name
Fig_name = [folder,'Along_track_temp_prof_glider_2models_',inst_name,'_',datestr(timeg(1),'mm-dd-yy'),'-',datestr(timeg(end),'mm-dd-yy')];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 
 
%%

%{
%% Figure with 3 models: GOFS3.1, GOFS3.0 and Copernicus

siz_title = 20;
siz_text = 16;

color = colormap(jet(length(latg)+1));
norm = (timeg-timeg(1))./(timeg(end)-timeg(1));
pos = round(norm.*length(latg))+1;

time_mat = repmat(timeg,1,size(tempg,1))';
time_vec = reshape(time_mat,1,size(time_mat,1)*size(time_mat,2));

depth_vec = reshape(presg,1,size(presg,1)*size(presg,2));
tempg_vec = reshape(tempg,1,size(tempg,1)*size(tempg,2));

marker.MarkerSize = 16;

figure
set(gcf,'position',[366 82 1134 902])

subplot(511)
set(gca,'position', [0.15 0.83 0.7181 0.1243])
contour(bath_lon,bath_lat,bath_elev',[0,-50,-100,-200,-1000,-2000,-4000,-8000],...
        'color','k','linewidth',2)
%contour(bath_lon,bath_lat,bath_elev',[0 0],'color','k','linewidth',2)
hold on
%contourf(bath_lon,bath_lat,bath_elev',[0 max(max(bath_elev))],'color',[0.5 0.2 0])
%contourf(bath_lon,bath_lat,bath_elev',[min(min(bath_elev)) -0.1])
axis equal
ylim(lat_lim)
xlim(lon_lim)
set(gca,'fontsize',siz_text)
t = title([inst_name,' ',' Track'],'fontsize',siz_title);
%set(t,'position',[-75.0000   38.35   -0.0000])
%set(t,'position',[-73.0000   40.6   -0.0000])

for i=1:length(latg)
    plot(long(i),latg(i),'.','markersize',12,'color',color(pos(i),:,:))
end
colormap('jet')
c = colorbar('v');
caxis([timeg(1) timeg(end)])
%timevec = datenum(date_ini):datenum(0,0,2,0,0,0):datenum(date_end);
timevec = timeg(1):datenum(0,0,10,0,0,0):timeg(end);
time_lab = datestr(timevec,'mmm/dd');
set(c,'ytick',timevec)
datetick(c,'keepticks')
set(c,'yticklabel',time_lab)

%%
subplot(512)
[h,c_h] = fast_scatter(time_vec',-depth_vec',tempg_vec','colorbar','vert','marker',marker);
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
xlabel('')
title(['Along track temperature profile ',inst_name],'fontsize',siz_title)
c = colorbar;
%c.Label.String = 'Potential Temperature (^oC)';
%c.Label.FontSize = siz_text;
colormap('jet')
caxis([min(min(temperature)) max(max(temperature))])
set(c,'ytick',round(min(min(temperature))):4:round(max(max(temperature))))
xlim([timeg(1) timeg(end)])
xticks(timeg(1):datenum(0,0,10):timeg(end))
datetick('x','keepticks')
set(gca,'xticklabel',{[]}) 
ylim([-max(depth_vec) 0])
set(gca,'xgrid','on','ygrid','on','layer','top')

%%
subplot(513)
pcolor(time31(oktime31),-depth31,target_temp31)
shading interp
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title('GOFS 3.1','fontsize',siz_title)
c = colorbar;
c.Label.String = 'Potential Temperature (^oC)';
c.Label.FontSize = siz_text;
%set(c,'ytick',min(min(temperature)):2:min(min(temperature)))
colormap('jet')
caxis([min(min(temperature)) max(max(temperature))])
set(c,'ytick',round(min(min(temperature))):4:round(max(max(temperature))))
xlim([timeg(1) timeg(end)])
xticks(timeg(1):datenum(0,0,10):timeg(end))
datetick('x','keepticks')
set(gca,'xticklabel',{[]})
ylim([-max(depth_vec) 0])
set(gca,'xgrid','on','ygrid','on','layer','top')

%%
subplot(514)
pcolor(time30(oktime30),-depth30,target_temp30)
shading interp
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title('GOFS 3.0','fontsize',siz_title)
c = colorbar;
%c.Label.String = 'Potential Temperature (^oC)';
%c.Label.FontSize = siz_text;
%set(c,'ytick',min(min(temperature)):4:min(min(temperature)))
colormap('jet')
caxis([min(min(temperature)) max(max(temperature))])
set(c,'ytick',round(min(min(temperature))):4:round(max(max(temperature))))
xlim([timeg(1) timeg(end)])
xticks(timeg(1):datenum(0,0,10):timeg(end))
datetick('x','keepticks')
set(gca,'xticklabel',{[]})
ylim([-max(depth_vec) 0])
set(gca,'xgrid','on','ygrid','on','layer','top')

%%
subplot(515)
%pcolor([timecop(2);timecop(oktimecop)],-depthcop,target_tempcop)
pcolor(timecop(oktimecop),-depthcop,target_tempcop)
shading interp
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
title('Copernicus','fontsize',siz_title)
c = colorbar;
%c.Label.String = 'Potential Temperature (^oC)';
%c.Label.FontSize = siz_text;
%set(c,'ytick',min(min(temperature)):2:min(min(temperature)))
colormap('jet')
caxis([min(min(temperature)) max(max(temperature))])
set(c,'ytick',round(min(min(temperature))):4:round(max(max(temperature))))
xlim([timeg(1) timeg(end)])
xticks(timeg(1):datenum(0,0,10):timeg(end))
datetick('x','keepticks')
ylim([-max(depth_vec) 0])
set(gca,'position',[0.1300    0.1100    0.735    0.1243])
set(c,'position',[0.872    0.1098    0.0141    0.1242])
set(gca,'xgrid','on','ygrid','on','layer','top')

%%
% Figure name
Fig_name = [folder,'Along_track_temp_prof_glider_3models_',inst_name,'_',plat_type];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 
%} 

end
           
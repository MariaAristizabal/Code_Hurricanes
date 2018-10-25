%% Figure with 3 models

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
t = title([inst_name,' ',plat_type,' Track'],'fontsize',siz_title);
%set(t,'position',[-75.0000   38.35   -0.0000])
%set(t,'position',[-73.0000   40.6   -0.0000])

for i=1:length(latg)
    plot(long(i),latg(i),'.','markersize',12,'color',color(pos(i),:,:))
end

%%
colormap('jet')
c = colorbar('v');
caxis([timeg(1) timeg(end)])
timevec = datenum(date_ini):datenum(0,0,10,0,0,0):datenum(date_end);
time_lab = datestr(timevec,'mmm/dd');
set(c,'ytick',timevec)
datetick(c,'keepticks')
set(c,'yticklabel',time_lab)
%%
ylim([25 30])
xlim([-90 -80])

%%
subplot(512)
[h,c_h] = fast_scatter(time_vec',-depth_vec',tempg_vec','colorbar','vert','marker',marker);
set(gca,'fontsize',siz_text)
ylabel('Depth (m)')
xlabel('')
title(['Along track temperature profile ',inst_name,' ',plat_type],'fontsize',siz_title)
c = colorbar;
%c.Label.String = 'Potential Temperature (^oC)';
%c.Label.FontSize = siz_text;
colormap('jet')
caxis([min(min(temperature)) max(max(temperature))])
set(c,'ytick',round(min(min(temperature))):2:round(max(max(temperature))))
xlim([datenum(date_ini) datenum(date_end)])
xticks(datenum(date_ini):datenum(0,0,5):datenum(date_end))
datetick('x','keepticks')
set(gca,'xticklabel',{[]}) 
ylim([-max(depth_vec) 0])
grid on

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
set(c,'ytick',min(min(temperature)):2:min(min(temperature)))
colormap('jet')
caxis([min(min(temperature)) max(max(temperature))])
set(c,'ytick',round(min(min(temperature))):2:round(max(max(temperature))))
xlim([datenum(date_ini) datenum(date_end)])
xticks(datenum(date_ini):datenum(0,0,5):datenum(date_end))
datetick('x','keepticks')
set(gca,'xticklabel',{[]})
ylim([-max(depth_vec) 0])
grid on

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
set(c,'ytick',min(min(temperature)):2:min(min(temperature)))
colormap('jet')
caxis([min(min(temperature)) max(max(temperature))])
set(c,'ytick',round(min(min(temperature))):2:round(max(max(temperature))))
xlim([datenum(date_ini) datenum(date_end)])
xticks(datenum(date_ini):datenum(0,0,5):datenum(date_end))
datetick('x','keepticks')
set(gca,'xticklabel',{[]})
ylim([-max(depth_vec) 0])
grid on
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
set(c,'ytick',min(min(temperature)):2:min(min(temperature)))
colormap('jet')
caxis([min(min(temperature)) max(max(temperature))])
set(c,'ytick',round(min(min(temperature))):2:round(max(max(temperature))))
xlim([datenum(date_ini) datenum(date_end)])
xticks(datenum(date_ini):datenum(0,0,5):datenum(date_end))
datetick('x','keepticks')
ylim([-max(depth_vec) 0])
set(gca,'position',[0.1300    0.1100    0.735    0.1243])
set(c,'position',[0.872    0.1098    0.0141    0.1242])
grid on
%%
% Figure name
Fig_name = [folder,'Along_track_temp_prof_glider_3models_',inst_name,'_',plat_type,'_'];
wysiwyg
print([Fig_name,'.png'],'-dpng','-r300') 


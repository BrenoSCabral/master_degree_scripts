

def curr_window(dep_sup, dep_bot, lon_e, lon_d, top_e, top_d, bot_e, bot_d,
                path,ticks, l_contour, bathy_e, bathy_d, cmap='cool_r', cbar_label='', cbar_ticks=[]):

    fig, ax = plt.subplots(2,2, figsize=(12,9))


    im1 = ax[0,0].contourf(lon_e, -dep_sup, top_e, cmap=cmap, levels=ticks)
    

    # ax[1,0] = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree()) 
    
    # ax[1,0] = subplot_map(ax[1,0], s)


    # Obtém a posição do eixo original para sobrepor o GeoAxes
    left, bottom, width, height = ax[1,0].get_position().bounds
    left = left + .03
    bottom = bottom +.04
    height = height -.04
    width = width - .03

    # Cria um novo GeoAxes com a mesma posição e projeção geográfica
    ax_geo = fig.add_axes(
        [left, bottom, width, height],
        projection=ccrs.PlateCarree()  # Projeção desejada
    )


    # Define o fundo do GeoAxes como transparente
    # ax_geo.patch.set_alpha(0)  # Importante para ver o eixo original
    ax_geo.set_xticks([])
    ax_geo.set_yticks([])


    ax_geo = subplot_map(ax_geo, s)

    ax_geo.set_zorder(2)  # Coloca o mapa acima do eixo original
    ax[1,0].set_zorder(1)  # Mantém o eixo original atrás

    ax[0,0].set_facecolor([0,0,0,0.6])

    if cmap == 'bwr':
        contour_color = 'black'
    else:
        contour_color = 'lightgrey'

    if l_contour[int(len(l_contour)/2-.5)] == 0:
        l_contour = np.delete(l_contour, int(len(l_contour)/2-.5))
        ax[0,0].contour(lon_e, -dep_sup, top_e, colors=contour_color ,linewidths=1, levels=np.asarray([0]))
        ax[0,1].contour(lon_d, -dep_sup, top_d, colors=contour_color ,linewidths=1, levels=np.asarray([0]))
        ax[1,1].contour(lon_d, -dep_bot, bot_d, colors=contour_color ,linewidths=1, levels=np.asarray([0]))


    elif len(l_contour) == 2:
        try:
            if np.nanmin(top_e) <20 and np.nanmax(top_e) > 20:
                cs0 = ax[0,0].contour(lon_e, -dep_sup, np.array(top_e).T, colors='white',linewidths=1, levels=[20])
                ax[0,0].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro ES %')

        try:
            if np.nanmin(top_d) <20 and np.nanmax(top_d) > 20:
                cs0 = ax[0,1].contour(lon_d, -dep_sup, np.array(top_d).T, colors='white',linewidths=1, levels=[20])
                ax[0,1].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro DS %')


        try:
            if np.nanmin(bot_d) <20 and np.nanmax(bot_d) > 20:
                cs0 = ax[1,1].contour(lon_d, -dep_bot, np.array(bot_d).T, colors='white',linewidths=1, levels=[20])
                ax[1,1].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro DI %')



        try:
            if np.nanmin(top_e) <50 and np.nanmax(top_e) > 50:
                cs0 = ax[0,0].contour(lon_e, -dep_sup, np.array(top_e).T, colors='black',linewidths=1, levels=[50])
                ax[0,0].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro ES %')

        try:
            if np.nanmin(top_d) <50 and np.nanmax(top_d) > 50:
                cs0 = ax[0,1].contour(lon_d, -dep_sup, np.array(top_d).T, colors='black',linewidths=1, levels=[50])
                ax[0,1].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro DS %')


        try:
            if np.nanmin(bot_d) <50 and np.nanmax(bot_d) > 50:
                cs0 = ax[1,1].contour(lon_d, -dep_bot, np.array(bot_d).T, colors='black',linewidths=1, levels=[50])
                ax[1,1].clabel(cs0, inline=True,fontsize=10)
        except:
            print('erro DI %')




        # l_contour = np.array([])


    else:

        cs1 = ax[0,0].contour(lon_e, -dep_sup, top_e, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
        ax[0,0].clabel(cs1, inline=True,fontsize=10)
    
        cs1 = ax[0,1].contour(lon_d, -dep_sup, top_d, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
        ax[0,1].clabel(cs1, inline=True,fontsize=10)

        cs2 = ax[1,1].contour(lon_d, - dep_bot, bot_d, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
        ax[1,1].clabel(cs2, inline=True,fontsize=10)


    ax[0,0].set_xticks([])


    ## Direita superior -----------------------------------



    # im1 = ax[0,1].contourf(lon_d, -dep_sup, np.array(top_d).T, cmap=cmap, vmin=ticks[0], vmax=ticks[-1],
    # shading='gouraud')
    im1 = ax[0,1].contourf(lon_d, -dep_sup, top_d, cmap=cmap, levels=ticks)
    ax[0,1].set_facecolor([0,0,0,0.6]) 


    ax[0,1].set_yticks([])
    ax[0,1].set_xticks([])

    # cs2 = ax[1,0].contour(lon_e, -dep_bot, np.array(bot_e).T, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
    # ax[1,0].clabel(cs2, inline=True,fontsize=10)


    # ax[1,0].set_ylabel("Depth (m)")

    ax[1,0].set_facecolor([0,0,0,0.6])


    ## Esquerda Inferior -----------------------------------


    im2 = ax[1,0].contourf(lon_e, -dep_bot, bot_e, cmap=cmap, levels=ticks)
    ax[1,0].set_facecolor([0,0,0,0.6]) 

    try:
        cs2 = ax[1,0].contour(lon_e, -dep_bot, bot_e, colors=contour_color, linestyles= 'dashed',linewidths=0.5, levels=l_contour)
        ax[1,0].clabel(cs2, inline=True,fontsize=10)
    except Exception as e:
        print('deu ruim no cs2')


    # ax[1,0].set_ylabel("Depth (m)")




    ## Direita Inferior -----------------------------------



    # im2 = ax[1,1].pcolormesh(lon_d, - dep_bot, np.array(bot_d).T, cmap=cmap, vmin=ticks[0], vmax=ticks[-1], shading='gouraud')
    im2 = ax[1,1].contourf(lon_d, - dep_bot, bot_d, cmap=cmap, levels=ticks)
    ax[1,1].set_facecolor([0,0,0,0.6]) 




    ax[1,1].set_yticks([])

    # Adicionar colorbar à direita
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im1, cax=cbar_ax, extend='both', ticks=cbar_ticks)
    cbar.set_label(cbar_label)

    plt.subplots_adjust(wspace=0.03, hspace=0.05)

    fig.text(0.05, 0.5, 'Depth (m)', ha='center', va='center', rotation='vertical', fontsize=12)

    fig.text(0.5, 0.05, 'Distance from the coast (km)', ha='center', va='center', rotation='horizontal', fontsize=12)




    ax[0][0].plot(lon_e, bathy_e.data,
                  'r--')
    

    ax[0][1].plot(lon_d, bathy_d.data,
                'r--')
    

    ax[1][1].plot(lon_d, bathy_d.data,
                  'r--')

    
    ax[1][0].plot(lon_e, bathy_e.data,
                  'r--')
    
    sup_lim = (-dep_sup[-1], -dep_sup[0])
    bot_lim = (-dep_bot[-1], -dep_bot[0])




    ax[0][0].set_ylim(sup_lim)
    
    ax[1][0].set_ylim(bot_lim)

    ax[0][1].set_ylim(sup_lim)
    ax[1][1].set_ylim(bot_lim)

    # plt.savefig('/Users/breno/mestrado/tudo_adj_t1.png')
    plt.show()


l_contour = np.array([20, 50])
cbar_ticks = np.arange(0, 101, 10)


perc_top_e = (varf_top_e/var_top_e) * 100
perc_top_d = (varf_top_d/var_top_d) * 100
perc_bot_e = (varf_bot_e/var_bot_e) * 100
perc_bot_d = (varf_bot_d/var_bot_d) * 100



curr_window(dep_sup=d_top, dep_bot=d_bot, lon_e=dist_km_e, lon_d=dist_km_d,
            top_e=perc_top_e, top_d=perc_top_d, bot_e=perc_bot_e, bot_d=perc_bot_d,
            path=output_dir + f'/perc_{s}', ticks=np.arange(0,100,.1), l_contour=l_contour,
            cmap='inferno', cbar_label='%', cbar_ticks=cbar_ticks,
            bathy_e=bathy_interp[:div_ref], bathy_d=bathy_interp[div_ref:])



fig = plt.figure(figsize=(14,8))
fig.suptitle('series filtradas de ssh')
ax = fig.add_subplot(111)
ax.spines['top'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.spines['left'].set_color('none')
ax.spines['right'].set_color('none')
ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)

ax1= fig.add_subplot(411)
ax2= fig.add_subplot(412)
ax3= fig.add_subplot(413)
ax4= fig.add_subplot(414)

axes = [ax1, ax2, ax3, ax4]

time = pd.date_range(f'{2012}-01-01', f'{2012}-12-31', freq='D')
time = time[30:]
for serie , axe in zip(series, axes):
    data = series[serie]
    data = data[30:]
    if serie == 'Fortaleza_2012.csv':
        data = data*100
    if serie == 'TEPORTIm_2012.csv':
        time = time[152:]
    axe.plot(time, data, label=serie)
    axe.grid()
    axe.legend(fontsize='x-small', loc='lower left')
    axe.set_xlim([datetime.date(2012, 1, 1), datetime.date(2012, 12, 31)])

ax.set_ylabel('(cm)')
# plt.show()
plt.tight_layout()

plt.savefig('/Users/breno/Documents/Mestrado/resultados/2012/figs/series/' + 'compare_time_series', dpi=200)
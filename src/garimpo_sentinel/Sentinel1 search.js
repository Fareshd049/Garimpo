// Sentinel-1 scene search by date over an AOI (Google Earth Engine)

// Parameters
var AOI = ee.Geometry.Rectangle([-61.6, -6.2, -60.7, -5.3]);
var TARGET_DATE = '2024-08-02';
var WINDOW_DAYS = 1;

var VIS_PARAMS = {min: -25, max: 0};
var EXPORT_SCALE = 10;
var THUMB_SIZE = 1024;

Map.centerObject(AOI, 11);

// Search window around the target date
var targetDate = ee.Date(TARGET_DATE);
var startDate = targetDate.advance(-WINDOW_DAYS, 'day');
var endDate = targetDate.advance(WINDOW_DAYS, 'day');

// Sentinel-1 GRD, IW mode, VV polarisation
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(AOI)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filterDate(startDate, endDate)
  .sort('system:time_start');

print('S1 scenes found near ' + TARGET_DATE + ':', s1.size());
print('Scene list:', s1);

// First scene found
var s1Image = ee.Image(s1.first());
print('Selected scene date:', s1Image.date());

var vv = s1Image.select('VV').clip(AOI);
Map.addLayer(vv, VIS_PARAMS, 'Sentinel-1 VV');

// Download links
var pngUrl = vv.visualize(VIS_PARAMS).getThumbURL({
  region: AOI,
  dimensions: THUMB_SIZE,
  format: 'png'
});
print('PNG:', pngUrl);

var tiffUrl = vv.getDownloadURL({
  region: AOI,
  scale: EXPORT_SCALE,
  format: 'GEO_TIFF'
});
print('GeoTIFF:', tiffUrl);
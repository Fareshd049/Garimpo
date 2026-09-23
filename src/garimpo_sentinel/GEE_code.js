// PROJET : Détection garimpo - Rio Madeira (Manicoré)
// Conditions : écart S1<->S2 <= 6j, nuages < 30% sur l'AOI,
// intersection spatiale réelle S1∩S2∩AOI, avec seuil de pixels minimum.

var aoi = ee.Geometry.Rectangle([-61.6, -6.2, -60.7, -5.3]);
var DATE_DEBUT = '2024-01-01';
var DATE_FIN   = '2024-12-31';

var FENETRE_JOURS     = 6;
var SEUIL_NUAGES_AOI  = 0.30;
var SEUIL_PROBA_PIXEL = 40;
var SEUIL_PIXELS_MIN  = 200000; // ~1.4 x 1.4 km à 10m

Map.centerObject(aoi, 9);
Map.addLayer(aoi, {color: 'white'}, 'AOI');

// --- Collections de base ---
var s1_collection = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi).filterDate(DATE_DEBUT, DATE_FIN)
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
  .select(['VV', 'VH']);

var s2_sr = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi).filterDate(DATE_DEBUT, DATE_FIN)
  .select(['B4', 'B3', 'B2', 'B8']);

var s2_cloud_proba = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
  .filterBounds(aoi).filterDate(DATE_DEBUT, DATE_FIN);

// Attache à chaque S2 sa carte de probabilité de nuage
var joint = ee.Join.saveFirst('nuage_proba').apply({
  primary: s2_sr, secondary: s2_cloud_proba,
  condition: ee.Filter.equals({leftField: 'system:index', rightField: 'system:index'})
});
var s2_collection = ee.ImageCollection(joint).filter(ee.Filter.notNull(['nuage_proba']));

print('Nb images S1:', s1_collection.size());
print('Nb images S2:', s2_collection.size());

// --- Jointure temporelle (<= 6 jours) ---
var filtreTemps = ee.Filter.maxDifference({
  difference: FENETRE_JOURS * 86400000,
  leftField: 'system:time_start', rightField: 'system:time_start'
});
var jointure = ee.Join.saveAll({matchesKey: 'candidats_s2'});
var s1_avec_s2 = ee.ImageCollection(jointure.apply(s1_collection, s2_collection, filtreTemps));

var n_s1 = s1_avec_s2.size().getInfo();
print('Nb images S1 avec un candidat S2 dans la fenêtre:', n_s1);

// Aplatit toutes les paires candidates (métadonnées seulement)
var s1_liste = s1_avec_s2.toList(n_s1);
var candidats_par_s1 = s1_liste.map(function(elem) {
  var s1_img = ee.Image(elem);
  var date_s1 = s1_img.date();
  var s1_index = s1_img.get('system:index');
  var candidats = ee.List(s1_img.get('candidats_s2'));

  var features = candidats.map(function(s2_elem) {
    var s2_img = ee.Image(s2_elem);
    var ecart = s2_img.date().millis().subtract(date_s1.millis()).abs();
    return ee.Feature(null, {
      s1_index: s1_index,
      s2_index: s2_img.get('system:index'),
      date_s1: date_s1.format('YYYY-MM-dd'),
      date_s2: s2_img.date().format('YYYY-MM-dd'),
      ecart_ms: ecart
    });
  });
  return ee.FeatureCollection(features);
});
var candidats_plats = ee.FeatureCollection(candidats_par_s1).flatten();

// --- Nuages mesurés sur l'AOI (pas la métadonnée de scène entière) ---
function mesurerNuage(feature) {
  var s2_index = feature.get('s2_index');
  var s2_img = ee.Image(s2_collection.filter(ee.Filter.eq('system:index', s2_index)).first());
  var proba = ee.Image(s2_img.get('nuage_proba')).select('probability');
  var pct = proba.gt(SEUIL_PROBA_PIXEL).reduceRegion({
    reducer: ee.Reducer.mean(), geometry: aoi, scale: 60, maxPixels: 1e9, bestEffort: true
  }).get('probability');
  return feature.set('pct_nuage_aoi', pct);
}

var candidats_valides = candidats_plats.map(mesurerNuage)
  .filter(ee.Filter.lt('pct_nuage_aoi', SEUIL_NUAGES_AOI));

print('Nb paires respectant temps + nuages AOI:', candidats_valides.size());

// Une seule paire par image S1 : la plus proche en temps parmi les valides
var toutes = candidats_valides.getInfo().features.map(function(f) { return f.properties; });
var meilleurs = {};
toutes.forEach(function(c) {
  if (!meilleurs[c.s1_index] || c.ecart_ms < meilleurs[c.s1_index].ecart_ms) {
    meilleurs[c.s1_index] = c;
  }
});
var pairesFinales = Object.keys(meilleurs).map(function(k) { return meilleurs[k]; });
print('Nb paires finales candidates:', pairesFinales.length);

// --- Intersection spatiale + seuil de pixels + export ---
var maxError = 10;
var scale_export = 10;
var couleurs = ['red','blue','green','orange','purple','cyan','magenta','yellow','brown','lime'];
var nb_exportees = 0;

for (var i = 0; i < pairesFinales.length; i++) {
  var p = pairesFinales[i];

  // Retrouve S1 et son S2 via la jointure d'origine (garde le lien exact)
  var s1_img = ee.Image(s1_avec_s2.filter(ee.Filter.eq('system:index', p.s1_index)).first());
  var candidats_de_s1 = ee.ImageCollection(ee.List(s1_img.get('candidats_s2')));
  var s2_img = ee.Image(candidats_de_s1.filter(ee.Filter.eq('system:index', p.s2_index)).first());

  // Zone réellement commune aux deux capteurs, à l'intérieur de l'AOI
  var zone_utile = aoi
    .intersection(s1_img.geometry(), maxError)
    .intersection(s2_img.geometry(), maxError);

  var aire_zone = zone_utile.area(maxError).getInfo();
  var pixels_utiles = aire_zone / (scale_export * scale_export);

  if (pixels_utiles < SEUIL_PIXELS_MIN) {
    print('Paire ignorée (' + Math.round(pixels_utiles) + ' px):', 'garimpo_S1_' + p.date_s1 + '_S2_' + p.date_s2);
    continue;
  }

  var fusion = s1_img.addBands(s2_img)
    .rename(['VV', 'VH', 'Rouge', 'Vert', 'Bleu', 'NIR'])
    .clip(zone_utile).toFloat();

  var nom = 'garimpo_S1_' + p.date_s1 + '_S2_' + p.date_s2;

  // Affiche la zone de CHAQUE paire exportée, avec une couleur différente
  var couleur = couleurs[nb_exportees % couleurs.length];
  Map.addLayer(zone_utile, {color: couleur}, 'Zone ' + nom);

  Export.image.toDrive({
    image: fusion,
    description: nom,
    folder: 'garimpo_dataset',
    fileNamePrefix: nom,
    region: aoi,
    scale: 10,
    crs: 'EPSG:32720',
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF'
  });

  nb_exportees++;
  print('Tâche créée (' + nb_exportees + '):', nom);
}

print('✅ ' + nb_exportees + ' tâches créées → onglet Tasks → clique RUN sur chacune.');
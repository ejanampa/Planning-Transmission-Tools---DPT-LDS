import os
import io
import zipfile
import tempfile
import xml.etree.ElementTree as ET
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from shapely.geometry import box

st.set_page_config(layout="wide", page_title="Mapa de Densidad Eléctrica Multianual")
st.title("⚡ Mapa de Densidad de Carga Eléctrica (MW/km²)")

# --- CAMBIO CLAVE PARA STREAMLIT CLOUD ---
# Busca los archivos en el propio directorio del repositorio en lugar de una ruta absoluta local (D:\...)
PATH_DIR = os.path.dirname(os.path.abspath(__file__))

FILE_EXCEL = "GuiaSED_Dic2025.xlsx"
FILE_GPKG = "Radio SETs.gpkg"
HOJA_EXCEL = "SED"

CRS_UTM_18S = "EPSG:32718"

# ---------------------------------------------------------
# 1. FUNCIONES AUXILIARES DE COLOR Y RANGO
# ---------------------------------------------------------
def obtener_color_densidad(d):
    if d >= 4.0:
        return '#FF0000'  # Muy Alta - Rojo
    elif d >= 2.5:
        return '#FF8C00'  # Alta - Anaranjado
    elif d >= 1.5:
        return '#0000FF'  # Media - Azul
    elif d >= 0.25:
        return '#008000'  # Baja - Verde
    else:
        return '#FFFF00'  # Muy Baja - Amarillo

def obtener_rango_densidad(d):
    if d >= 4.0:
        return 'Muy Alta (d >= 4.0)'
    elif d >= 2.5:
        return 'Alta (4.0 > d >= 2.5)'
    elif d >= 1.5:
        return 'Media (2.5 > d >= 1.5)'
    elif d >= 0.25:
        return 'Baja (1.5 > d >= 0.25)'
    else:
        return 'Muy Baja (d < 0.25)'

def hex_to_kml_color(hex_str, alpha="aa"):
    hex_str = hex_str.lstrip('#')
    r, g, b = hex_str[0:2], hex_str[2:4], hex_str[4:6]
    return f"{alpha}{b}{g}{r}"

# ---------------------------------------------------------
# 2. CARGA Y PROCESAMIENTO DE DATOS
# ---------------------------------------------------------
@st.cache_data
def cargar_y_procesar_datos_base(folder_path):
    excel_path = os.path.join(folder_path, FILE_EXCEL)
    gpkg_path = os.path.join(folder_path, FILE_GPKG)

    if not os.path.exists(excel_path) or not os.path.exists(gpkg_path):
        st.error(f"No se encontraron los archivos requeridos ({FILE_EXCEL} / {FILE_GPKG}) en el repositorio.")
        return None, None

    gdf_concesion = gpd.read_file(gpkg_path)
    if gdf_concesion.crs != CRS_UTM_18S:
        gdf_concesion_utm = gdf_concesion.to_crs(CRS_UTM_18S)
    else:
        gdf_concesion_utm = gdf_concesion.copy()

    df_nodos = pd.read_excel(excel_path, sheet_name=HOJA_EXCEL)

    df_nodos['X_clean'] = (
        df_nodos['X Ubicación Lds WGS 84 Utm']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .str.replace(r'[^0-9.]', '', regex=True)
        .astype(float)
    )
    df_nodos['Y_clean'] = (
        df_nodos['Y Ubicación Lds WGS 84 Utm']
        .astype(str)
        .str.replace(',', '.', regex=False)
        .str.replace(r'[^0-9.]', '', regex=True)
        .astype(float)
    )

    gdf_nodos_utm = gpd.GeoDataFrame(
        df_nodos,
        geometry=gpd.points_from_xy(df_nodos['X_clean'], df_nodos['Y_clean']),
        crs=CRS_UTM_18S
    )

    return gdf_concesion_utm, gdf_nodos_utm

@st.cache_data
def obtener_grilla_y_matriz_espacial(_gdf_concesion_utm, _gdf_nodos_utm, step=1000):
    xmin, ymin, xmax, ymax = _gdf_concesion_utm.total_bounds
    cols = np.arange(xmin, xmax, step)
    rows = np.arange(ymin, ymax, step)

    polygons = [box(x, y, x + step, y + step) for x in cols for y in rows]
    grid = gpd.GeoDataFrame({'geometry': polygons}, crs=_gdf_concesion_utm.crs)
    grid_clipped = gpd.clip(grid, _gdf_concesion_utm)
    grid_clipped['grid_id'] = grid_clipped.index.astype(str)

    joined = gpd.sjoin(_gdf_nodos_utm, grid_clipped[['grid_id', 'geometry']], how='inner', predicate='within')

    return grid_clipped, joined

# ---------------------------------------------------------
# 3. FUNCIONES DE EXPORTACIÓN
# ---------------------------------------------------------
@st.cache_data
def exportar_pdf(_gdf_grid, _gdf_concesion, col_demanda):
    fig, ax = plt.subplots(figsize=(8, 8))

    _gdf_concesion.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1.2, zorder=1)

    grid_pdf = _gdf_grid.copy()
    grid_pdf['color_hex'] = grid_pdf['Densidad_MW_km2'].apply(obtener_color_densidad)

    for hex_c, group in grid_pdf.groupby('color_hex'):
        group.plot(ax=ax, facecolor=hex_c, edgecolor='#444444', linewidth=0.3, alpha=0.65, zorder=2)

    legend_elements = [
        mpatches.Patch(facecolor='#FF0000', edgecolor='gray', alpha=0.7, label='Muy Alta (d ≥ 4.0)'),
        mpatches.Patch(facecolor='#FF8C00', edgecolor='gray', alpha=0.7, label='Alta (4.0 > d ≥ 2.5)'),
        mpatches.Patch(facecolor='#0000FF', edgecolor='gray', alpha=0.7, label='Media (2.5 > d ≥ 1.5)'),
        mpatches.Patch(facecolor='#008000', edgecolor='gray', alpha=0.7, label='Baja (1.5 > d ≥ 0.25)'),
        mpatches.Patch(facecolor='#FFFF00', edgecolor='gray', alpha=0.7, label='Muy Baja (d < 0.25)'),
        mpatches.Patch(facecolor='none', edgecolor='black', linewidth=1.2, label='Límite Concesión')
    ]

    ax.legend(
        handles=legend_elements, 
        loc='lower left', 
        title="Densidad (MW/km²)", 
        title_fontsize=8,
        fontsize=7.5, 
        frameon=True, 
        facecolor='white', 
        framealpha=0.85,
        handletextpad=0.4,
        labelspacing=0.3
    )

    ax.set_title(f"Mapa de Densidad de Carga Eléctrica\nDemanda: {col_demanda}", fontsize=10, fontweight='bold', pad=8)
    ax.set_xlabel("UTM Este (m)", fontsize=8)
    ax.set_ylabel("UTM Norte (m)", fontsize=8)

    ax.ticklabel_format(style='plain', useOffset=False)
    ax.tick_params(
        axis='both', 
        which='major', 
        labelsize=7.5,
        top=False, 
        labeltop=False
    )
    ax.grid(True, linestyle='--', alpha=0.4)

    pdf_buffer = io.BytesIO()
    plt.savefig(pdf_buffer, format='pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)
    return pdf_buffer.getvalue()

@st.cache_data
def exportar_kmz(_gdf_grid, _gdf_nodos, col_demanda):
    grid_4326 = _gdf_grid.to_crs(epsg=4326)
    nodos_4326 = _gdf_nodos.to_crs(epsg=4326) if _gdf_nodos is not None and not _gdf_nodos.empty else None

    kml = ET.Element('kml', xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, 'Document')
    ET.SubElement(doc, 'name').text = f"Mapa de Densidad - {col_demanda}"

    styles = {
        'MuyAlta': '#FF0000',
        'Alta': '#FF8C00',
        'Media': '#0000FF',
        'Baja': '#008000',
        'MuyBaja': '#FFFF00'
    }

    for s_id, hex_c in styles.items():
        st_elem = ET.SubElement(doc, 'Style', id=s_id)
        ls = ET.SubElement(st_elem, 'LineStyle')
        ET.SubElement(ls, 'color').text = "ff333333"
        ET.SubElement(ls, 'width').text = "1"
        ps = ET.SubElement(st_elem, 'PolyStyle')
        ET.SubElement(ps, 'color').text = hex_to_kml_color(hex_c, alpha="aa")
        ET.SubElement(ps, 'fill').text = "1"
        ET.SubElement(ps, 'outline').text = "1"

    st_nodo = ET.SubElement(doc, 'Style', id="NodoStyle")
    is_nodo = ET.SubElement(st_nodo, 'IconStyle')
    ET.SubElement(is_nodo, 'color').text = "ffff0000"
    ET.SubElement(is_nodo, 'scale').text = "0.7"
    icon = ET.SubElement(is_nodo, 'Icon')
    ET.SubElement(icon, 'href').text = "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png"

    f_grid = ET.SubElement(doc, 'Folder')
    ET.SubElement(f_grid, 'name').text = f"Cuadrículas Densidad ({col_demanda})"
    ET.SubElement(f_grid, 'visibility').text = "0"

    for idx, row in grid_4326.iterrows():
        dens = row.get('Densidad_MW_km2', 0)
        if dens >= 4.0:
            s_url = '#MuyAlta'
        elif dens >= 2.5:
            s_url = '#Alta'
        elif dens >= 1.5:
            s_url = '#Media'
        elif dens >= 0.25:
            s_url = '#Baja'
        else:
            s_url = '#MuyBaja'

        pm = ET.SubElement(f_grid, 'Placemark')
        ET.SubElement(pm, 'name').text = f"Cuadrícula {row.get('grid_id', idx)}"
        ET.SubElement(pm, 'description').text = f"Densidad: {dens:.3f} MW/km²"
        ET.SubElement(pm, 'styleUrl').text = s_url
        ET.SubElement(pm, 'visibility').text = "0"

        geom = row.geometry
        polys = geom.geoms if geom.geom_type == 'MultiPolygon' else [geom]
        for poly in polys:
            if hasattr(poly, 'exterior'):
                poly_elem = ET.SubElement(pm, 'Polygon')
                outer = ET.SubElement(poly_elem, 'outerBoundaryIs')
                lr = ET.SubElement(outer, 'LinearRing')
                coords = " ".join([f"{pt[0]},{pt[1]},0" for pt in poly.exterior.coords])
                ET.SubElement(lr, 'coordinates').text = coords

    if nodos_4326 is not None:
        f_nodos = ET.SubElement(doc, 'Folder')
        ET.SubElement(f_nodos, 'name').text = "Nodos (SEDs)"
        ET.SubElement(f_nodos, 'visibility').text = "0"
        for idx, row in nodos_4326.iterrows():
            pm = ET.SubElement(f_nodos, 'Placemark')
            ET.SubElement(pm, 'name').text = str(row.get('numero', f'Nodo {idx}'))
            dem = row.get(col_demanda, 0)
            ET.SubElement(pm, 'description').text = f"Demanda ({col_demanda}): {dem} kW"
            ET.SubElement(pm, 'styleUrl').text = "#NodoStyle"
            ET.SubElement(pm, 'visibility').text = "0"

            geom = row.geometry
            if geom.geom_type == 'Point':
                pt_elem = ET.SubElement(pm, 'Point')
                ET.SubElement(pt_elem, 'coordinates').text = f"{geom.x},{geom.y},0"

    kml_bytes = ET.tostring(kml, encoding='utf-8', method='xml')
    kmz_buffer = io.BytesIO()
    with zipfile.ZipFile(kmz_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_bytes)
    return kmz_buffer.getvalue()

@st.cache_data
def exportar_shp_zip(_gdf_grid, _gdf_nodos, _gdf_concesion, col_demanda):
    buffer = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmpdir:
        grid_export = _gdf_grid.copy()
        grid_export['Dens_MWkm2'] = grid_export['Densidad_MW_km2'].round(3)
        grid_export['Rango_Dens'] = grid_export['Densidad_MW_km2'].apply(obtener_rango_densidad)
        grid_export['Color_HEX'] = grid_export['Densidad_MW_km2'].apply(obtener_color_densidad)
        grid_export.to_file(os.path.join(tmpdir, "cuadriculas_densidad.shp"), driver="ESRI Shapefile")

        if _gdf_nodos is not None and not _gdf_nodos.empty:
            nodos_export = _gdf_nodos[['numero', col_demanda, 'geometry']].copy()
            nodos_export = nodos_export.rename(columns={col_demanda: 'MD_kW'})
            nodos_export.to_file(os.path.join(tmpdir, "nodos_seds.shp"), driver="ESRI Shapefile")

        if _gdf_concesion is not None and not _gdf_concesion.empty:
            _gdf_concesion.to_file(os.path.join(tmpdir, "limite_concesion.shp"), driver="ESRI Shapefile")

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    zf.write(os.path.join(root, file), file)
    return buffer.getvalue()

@st.cache_data
def exportar_dxf(_gdf_grid, _gdf_nodos, _gdf_concesion):
    buffer = io.BytesIO()
    with tempfile.TemporaryDirectory() as tmpdir:
        _gdf_grid[['geometry']].to_file(os.path.join(tmpdir, "cuadriculas_densidad.dxf"), driver="DXF")
        if _gdf_nodos is not None and not _gdf_nodos.empty:
            _gdf_nodos[['geometry']].to_file(os.path.join(tmpdir, "nodos_seds.dxf"), driver="DXF")
        if _gdf_concesion is not None and not _gdf_concesion.empty:
            _gdf_concesion[['geometry']].to_file(os.path.join(tmpdir, "limite_concesion.dxf"), driver="DXF")

        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(tmpdir):
                for file in files:
                    zf.write(os.path.join(root, file), file)
    return buffer.getvalue()

# ---------------------------------------------------------
# 4. RENDERIZADO PRINCIPAL STREAMLIT
# ---------------------------------------------------------
gdf_concesion_utm, gdf_nodos_utm = cargar_y_procesar_datos_base(PATH_DIR)

if gdf_concesion_utm is not None and gdf_nodos_utm is not None:
    st.sidebar.header("⚙️ Selección de Demanda Anual")

    cols_demanda = [c for c in gdf_nodos_utm.columns if 'MD kW' in str(c)]
    if not cols_demanda:
        cols_demanda = [c for c in gdf_nodos_utm.columns if 'MD' in str(c) or 'kW' in str(c)]
    if not cols_demanda:
        cols_demanda = ['MD kW']

    col_seleccionada = st.sidebar.selectbox(
        "Seleccione la Demanda Anual:",
        options=cols_demanda,
        index=0
    )

    grid_utm, joined = obtener_grilla_y_matriz_espacial(gdf_concesion_utm, gdf_nodos_utm, step=1000)

    demanda_grilla = joined.groupby('grid_id')[col_seleccionada].sum().reset_index()
    demanda_grilla['Densidad_MW_km2'] = demanda_grilla[col_seleccionada] / 1000.0

    grid_resultado_utm = grid_utm.merge(demanda_grilla, on='grid_id', how='inner')
    grid_resultado_utm = grid_resultado_utm[grid_resultado_utm['Densidad_MW_km2'] > 0].copy()

    # --- SECCIÓN DE EXPORTACIÓN ---
    st.sidebar.markdown("---")
    st.sidebar.header("📥 Exportar Resultados")

    pdf_bytes = exportar_pdf(grid_resultado_utm, gdf_concesion_utm, col_seleccionada)
    st.sidebar.download_button(
        label="📄 Exportar a PDF (Mapa + Leyenda)",
        data=pdf_bytes,
        file_name=f"Mapa_Densidad_{col_seleccionada}.pdf",
        mime="application/pdf",
        key="btn_pdf"
    )

    kmz_bytes = exportar_kmz(grid_resultado_utm, gdf_nodos_utm, col_seleccionada)
    st.sidebar.download_button(
        label="🌍 Exportar a KMZ (Google Earth)",
        data=kmz_bytes,
        file_name=f"Mapa_Densidad_{col_seleccionada}.kmz",
        mime="application/vnd.google-earth.kmz",
        key="btn_kmz"
    )

    shp_bytes = exportar_shp_zip(grid_resultado_utm, gdf_nodos_utm, gdf_concesion_utm, col_seleccionada)
    st.sidebar.download_button(
        label="📦 Exportar a Shapefile (.zip)",
        data=shp_bytes,
        file_name=f"Mapa_Densidad_{col_seleccionada}_shp.zip",
        mime="application/zip",
        key="btn_shp"
    )

    dxf_bytes = exportar_dxf(grid_resultado_utm, gdf_nodos_utm, gdf_concesion_utm)
    st.sidebar.download_button(
        label="📐 Exportar a AutoCAD DXF (.zip)",
        data=dxf_bytes,
        file_name=f"Mapa_Densidad_{col_seleccionada}_dxf.zip",
        mime="application/zip",
        key="btn_dxf"
    )

    # Reproyectar a WGS84 para Folium
    grid_4326 = grid_resultado_utm.to_crs(epsg=4326)
    concesion_4326 = gdf_concesion_utm.to_crs(epsg=4326)
    nodos_4326 = gdf_nodos_utm.to_crs(epsg=4326)

    nodos_light = nodos_4326[['numero', col_seleccionada, 'geometry']].copy()
    nodos_light['Demanda_kW'] = nodos_light[col_seleccionada].round(2)

    centroide = concesion_4326.geometry.centroid.iloc[0]
    m = folium.Map(location=[centroide.y, centroide.x], zoom_start=12, tiles="CartoDB positron")

    folium.GeoJson(
        concesion_4326,
        style_function=lambda x: {'fillColor': 'none', 'color': 'black', 'weight': 2},
        name="Límite Concesión"
    ).add_to(m)

    def style_cuadrícula(feature):
        densidad = feature['properties']['Densidad_MW_km2']
        return {
            'fillColor': obtener_color_densidad(densidad),
            'color': '#333333',
            'weight': 0.7,
            'fillOpacity': 0.65
        }

    folium.GeoJson(
        grid_4326,
        style_function=style_cuadrícula,
        tooltip=folium.GeoJsonTooltip(
            fields=['grid_id', 'Densidad_MW_km2'],
            aliases=['Cuadrícula ID:', f'Densidad {col_seleccionada} (MW/km²):'],
            localize=True
        ),
        name=f"Densidad ({col_seleccionada})"
    ).add_to(m)

    folium.GeoJson(
        nodos_light,
        marker=folium.CircleMarker(radius=2.5, color="black", fill_color="cyan", fill_opacity=0.9, weight=0.4),
        tooltip=folium.GeoJsonTooltip(
            fields=['numero', 'Demanda_kW'],
            aliases=['Nodo:', 'Demanda (kW):']
        ),
        name="Nodos (SEDs)"
    ).add_to(m)

    Draw(
        export=True,
        filename='dibujo_interactivo.geojson',
        position='topleft',
        draw_options={'polyline': True, 'polygon': True, 'rectangle': True, 'marker': True, 'circle': False, 'circlemarker': False},
        edit_options={'edit': True}
    ).add_to(m)

    folium.LayerControl(position='topright').add_to(m)

    legend_html = '''
    <div style="
        position: fixed; 
        bottom: 25px; 
        left: 15px; 
        width: 200px; 
        background-color: #ffffff !important; 
        color: #000000 !important; 
        border: 2px solid #555555; 
        z-index: 99999; 
        font-size: 11px; 
        font-family: sans-serif;
        padding: 8px 10px; 
        border-radius: 6px;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.3);
    ">
    <b style="color: #000000 !important; font-size: 11px;">Rango Densidad (MW/km²)</b><br>
    <div style="margin-top: 4px; color: #000000 !important;">
        <i style="background:#FF0000; width: 12px; height: 12px; float: left; margin-right: 8px; border: 1px solid #333;"></i> <span style="color: #000000 !important;">Muy Alta (d ≥ 4.0)</span><br>
        <i style="background:#FF8C00; width: 12px; height: 12px; float: left; margin-right: 8px; border: 1px solid #333;"></i> <span style="color: #000000 !important;">Alta (4.0 > d ≥ 2.5)</span><br>
        <i style="background:#0000FF; width: 12px; height: 12px; float: left; margin-right: 8px; border: 1px solid #333;"></i> <span style="color: #000000 !important;">Media (2.5 > d ≥ 1.5)</span><br>
        <i style="background:#008000; width: 12px; height: 12px; float: left; margin-right: 8px; border: 1px solid #333;"></i> <span style="color: #000000 !important;">Baja (1.5 > d ≥ 0.25)</span><br>
        <i style="background:#FFFF00; width: 12px; height: 12px; float: left; margin-right: 8px; border: 1px solid #333;"></i> <span style="color: #000000 !important;">Muy Baja (d < 0.25)</span><br>
    </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    output = st_folium(
        m, 
        use_container_width=True, 
        height=680,
        returned_objects=["all_drawings"],
        key=f"mapa_{col_seleccionada}"
    )

    if output and output.get("all_drawings"):
        st.subheader("📌 Geometrías trazadas")
        st.json(output["all_drawings"])

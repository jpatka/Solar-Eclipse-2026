import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import CubicSpline
from astropy.time import Time
from astropy.coordinates import get_sun, get_body, ITRS, EarthLocation, solar_system_ephemeris
import astropy.units as u

# --- ŁADOWANIE PRECYZYJNYCH EFEMERYD JPL NASA ---
print("Ustawianie efemeryd NASA DE440 (może wymagać krótkiego pobrania danych przy pierwszym uruchomieniu)...")
solar_system_ephemeris.set('de440')

print("Generowanie symulacji wysokiej precyzji z pasem zaćmienia całkowitego...")

# --- 1. OBLICZANIE TRAJEKTORII I CENTRÓW CIENIA ---
NUM_STEPS = 1000  # Liczba kroków czasowych

time_start = Time('2026-08-12T16:45:00')
time_end = Time('2026-08-12T19:35:00')
time_steps = Time(time_start + np.linspace(0, (time_end - time_start).to(u.hour).value, NUM_STEPS) * u.hour)

centers_lon, centers_lat, time_labels = [], [], []
A_EARTH, B_EARTH = 6378137.0, 6356752.314245

for t in time_steps:
    sun_itrs = get_sun(t).transform_to(ITRS(obstime=t))
    moon_itrs = get_body("moon", t).transform_to(ITRS(obstime=t))

    r_sun = np.array([sun_itrs.x.to(u.m).value, sun_itrs.y.to(u.m).value, sun_itrs.z.to(u.m).value])
    r_moon = np.array([moon_itrs.x.to(u.m).value, moon_itrs.y.to(u.m).value, moon_itrs.z.to(u.m).value])

    v_dir = (r_moon - r_sun) / np.linalg.norm(r_moon - r_sun)

    A = (v_dir[0] ** 2 + v_dir[1] ** 2) / A_EARTH ** 2 + v_dir[2] ** 2 / B_EARTH ** 2
    B = 2 * ((r_moon[0] * v_dir[0] + r_moon[1] * v_dir[1]) / A_EARTH ** 2 + r_moon[2] * v_dir[2] / B_EARTH ** 2)
    C = (r_moon[0] ** 2 + r_moon[1] ** 2) / A_EARTH ** 2 + r_moon[2] ** 2 / B_EARTH ** 2 - 1

    discriminant = B ** 2 - 4 * A * C
    if discriminant >= 0:
        k = min((-B - np.sqrt(discriminant)) / (2 * A), (-B + np.sqrt(discriminant)) / (2 * A))
        p_intersect = r_moon + k * v_dir
        loc = EarthLocation.from_geocentric(p_intersect[0] * u.m, p_intersect[1] * u.m, p_intersect[2] * u.m)
        centers_lon.append(loc.lon.value)
        centers_lat.append(loc.lat.value)
        time_labels.append(t.datetime.strftime("%H:%M:%S UTC"))

# Wygładzenie linii centralnej zaćmienia całkowitego
t_arr = np.linspace(0, 1, len(centers_lon))
t_smooth = np.linspace(0, 1, 500)
path_lons = CubicSpline(t_arr, centers_lon)(t_smooth)
path_lats = CubicSpline(t_arr, centers_lat)(t_smooth)

CAMERA_LON = path_lons[len(path_lons) // 2]
CAMERA_LAT = path_lats[len(path_lats) // 2]


# --- 2. POMOCNICZE FUNKCJE GEOMETRYCZNE ---
def generate_spherical_circle(c_lon, c_lat, r_deg, points=120):
    angles = np.linspace(0, 2 * np.pi, points)
    r_rad = np.radians(r_deg)
    x = np.sin(r_rad) * np.cos(angles)
    y = np.sin(r_rad) * np.sin(angles)
    z = np.cos(r_rad) * np.ones_like(angles)

    theta = np.pi / 2 - np.radians(c_lat)
    x_r = x * np.cos(theta) + z * np.sin(theta)
    y_r = y
    z_r = -x * np.sin(theta) + z * np.cos(theta)

    phi = np.radians(c_lon)
    x_f = x_r * np.cos(phi) - y_r * np.sin(phi)
    y_f = x_r * np.sin(phi) + y_r * np.cos(phi)
    z_f = z_r

    return np.degrees(np.arctan2(y_f, x_f)), np.degrees(np.arcsin(np.clip(z_f, -1.0, 1.0)))


def clip_to_horizon(lons, lats, c_lon, c_lat, max_angle_deg=86.0):
    c_lon_rad, c_lat_rad = np.radians(c_lon), np.radians(c_lat)
    cam = np.array([np.cos(c_lat_rad) * np.cos(c_lon_rad), np.cos(c_lat_rad) * np.sin(c_lon_rad), np.sin(c_lat_rad)])
    max_rad = np.radians(max_angle_deg)
    clons, clats = [], []
    for lon, lat in zip(lons, lats):
        lon_r, lat_r = np.radians(lon), np.radians(lat)
        pt = np.array([np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r), np.sin(lat_r)])
        dot = np.clip(np.dot(pt, cam), -1.0, 1.0)
        if np.arccos(dot) > max_rad:
            proj = pt - dot * cam
            norm = np.linalg.norm(proj)
            if norm > 1e-6:
                pt_c = np.cos(max_rad) * cam + np.sin(max_rad) * (proj / norm)
                clons.append(np.degrees(np.arctan2(pt_c[1], pt_c[0])))
                clats.append(np.degrees(np.arcsin(np.clip(pt_c[2], -1.0, 1.0))))
        else:
            clons.append(lon)
            clats.append(lat)
    return clons, clats


# --- 3. GENEROWANIE PASA I IZOLINII W 3D ---
SHADOW_RADIUS_DEG = 36.0

lons_rad = np.radians(path_lons)
lats_rad = np.radians(path_lats)
X = np.cos(lats_rad) * np.cos(lons_rad)
Y = np.cos(lats_rad) * np.sin(lons_rad)
Z = np.sin(lats_rad)
P = np.vstack((X, Y, Z)).T

V = np.zeros_like(P)
V[1:-1] = P[2:] - P[:-2]
V[0] = P[1] - P[0]
V[-1] = P[-1] - P[-2]
V = V / np.linalg.norm(V, axis=1, keepdims=True)

N = np.cross(V, P)
N = N / np.linalg.norm(N, axis=1, keepdims=True)

# Odchylenia kątowe od osi centralnej
theta_100 = np.radians(1.35)  # Szerokość pasa zaćmienia całkowitego (~300 km)
theta_80 = np.radians(7.2)
theta_60 = np.radians(14.4)

P_100L = P * np.cos(theta_100) + N * np.sin(theta_100)
P_100R = P * np.cos(theta_100) - N * np.sin(theta_100)
P_80L = P * np.cos(theta_80) + N * np.sin(theta_80)
P_80R = P * np.cos(theta_80) - N * np.sin(theta_80)
P_60L = P * np.cos(theta_60) + N * np.sin(theta_60)
P_60R = P * np.cos(theta_60) - N * np.sin(theta_60)


def to_latlon(P_xyz):
    lon = np.degrees(np.arctan2(P_xyz[:, 1], P_xyz[:, 0]))
    lat = np.degrees(np.arcsin(np.clip(P_xyz[:, 2], -1.0, 1.0)))
    return list(lon), list(lat)


lon_100_l, lat_100_l = to_latlon(P_100L)
lon_100_r, lat_100_r = to_latlon(P_100R)
lon_80_l, lat_80_l = to_latlon(P_80L)
lon_80_r, lat_80_r = to_latlon(P_80R)
lon_60_l, lat_60_l = to_latlon(P_60L)
lon_60_r, lat_60_r = to_latlon(P_60R)

lon_100_l, lat_100_l = clip_to_horizon(lon_100_l, lat_100_l, CAMERA_LON, CAMERA_LAT)
lon_100_r, lat_100_r = clip_to_horizon(lon_100_r, lat_100_r, CAMERA_LON, CAMERA_LAT)
lon_80_l, lat_80_l = clip_to_horizon(lon_80_l, lat_80_l, CAMERA_LON, CAMERA_LAT)
lon_80_r, lat_80_r = clip_to_horizon(lon_80_r, lat_80_r, CAMERA_LON, CAMERA_LAT)
lon_60_l, lat_60_l = clip_to_horizon(lon_60_l, lat_60_l, CAMERA_LON, CAMERA_LAT)
lon_60_r, lat_60_r = clip_to_horizon(lon_60_r, lat_60_r, CAMERA_LON, CAMERA_LAT)

path_totality_lon = lon_100_l + lon_100_r[::-1]
path_totality_lat = lat_100_l + lat_100_r[::-1]

# --- 4. DATASET MIAST ---
cities = {
    'Reykjavík (100%)': [-21.94, 64.15],
    'Madryt (100%)': [-3.70, 40.42],
    'Palma de Mallorca (100%)': [2.65, 39.57],
    'Żagań (83%)': [15.31, 51.62],
    'Szczecin (84%)': [14.55, 53.43],
    'Gdańsk (82%)': [18.64, 54.35],
    'Warszawa (79%)': [21.01, 52.23],
    'Poznań (81%)': [16.92, 52.41],
    'Kraków (73%)': [19.94, 50.06],
    'Londyn (91%)': [-0.13, 51.51],
    'Paryż (92%)': [2.35, 48.86],
    'Berlin (87%)': [13.40, 52.52]
}

# --- 5. INICJALIZACJA INTERAKTYWNEGO GLOBU ---
init_p_lon, init_p_lat = generate_spherical_circle(centers_lon[0], centers_lat[0], SHADOW_RADIUS_DEG)
init_p_lon, init_p_lat = clip_to_horizon(init_p_lon, init_p_lat, CAMERA_LON, CAMERA_LAT)

fig = go.Figure()

fig.add_trace(go.Scattergeo(
    lon=path_totality_lon, lat=path_totality_lat,
    fill='toself', fillcolor='rgba(139, 0, 0, 0.35)',
    line=dict(color='darkred', width=1.5),
    name='Pas zaćmienia całkowitego (100%)'
))
fig.add_trace(go.Scattergeo(lon=init_p_lon, lat=init_p_lat, fill='toself', fillcolor='rgba(30, 30, 30, 0.40)',
                            line=dict(color='rgba(0,0,0,0.1)', width=1), name='Ruchomy półcień'))
fig.add_trace(
    go.Scattergeo(lon=[centers_lon[0]], lat=[centers_lat[0]], mode='markers', marker=dict(size=7, color='black'),
                  name='Środek cienia'))

fig.add_trace(go.Scattergeo(lon=path_lons, lat=path_lats, mode='lines',
                            line=dict(color='rgba(0,0,0,0.5)', width=1, dash='dashdot'), name='Oś środkowa zaćmienia'))
fig.add_trace(go.Scattergeo(lon=[0], lat=[90], mode='markers+text', marker=dict(size=8, color='darkblue', symbol='x'),
                            text=['Biegun Północny'], textposition='top center', name='Biegun N'))

fig.add_trace(go.Scattergeo(lon=lon_80_l, lat=lat_80_l, mode='lines',
                            line=dict(color='rgba(139, 0, 0, 0.6)', width=1.5, dash='dash'), name='Izolinia Max 80%',
                            legendgroup='80'))
fig.add_trace(go.Scattergeo(lon=lon_80_r, lat=lat_80_r, mode='lines',
                            line=dict(color='rgba(139, 0, 0, 0.6)', width=1.5, dash='dash'), showlegend=False,
                            legendgroup='80'))
fig.add_trace(go.Scattergeo(lon=lon_60_l, lat=lat_60_l, mode='lines',
                            line=dict(color='rgba(218, 165, 32, 0.7)', width=1.5, dash='dash'), name='Izolinia Max 60%',
                            legendgroup='60'))
fig.add_trace(go.Scattergeo(lon=lon_60_r, lat=lat_60_r, mode='lines',
                            line=dict(color='rgba(218, 165, 32, 0.7)', width=1.5, dash='dash'), showlegend=False,
                            legendgroup='60'))

fig.add_trace(go.Scattergeo(
    lon=[c[0] for c in cities.values()], lat=[c[1] for c in cities.values()],
    mode='markers+text', marker=dict(size=7, color='indigo', symbol='circle'),
    text=list(cities.keys()), textposition='top center',
    textfont=dict(size=10, family="Arial Bold", color="black"), name='Kluczowe miasta'
))

# --- 6. KLATKI ANIMACJI ---
frames = []
for i in range(len(centers_lon)):
    p_lon, p_lat = generate_spherical_circle(centers_lon[i], centers_lat[i], SHADOW_RADIUS_DEG)
    p_lon, p_lat = clip_to_horizon(p_lon, p_lat, CAMERA_LON, CAMERA_LAT)
    frames.append(go.Frame(
        data=[
            go.Scattergeo(lon=path_totality_lon, lat=path_totality_lat),
            go.Scattergeo(lon=p_lon, lat=p_lat),
            go.Scattergeo(lon=[centers_lon[i]], lat=[centers_lat[i]])
        ],
        name=f"frame_{i}"
    ))
fig.frames = frames

# --- 7. PANEL STEROWANIA I USTAWIENIA WIDOKU ---
updatemenus = [dict(
    type="buttons", direction="left", showactive=False, x=0.08, y=0, xanchor="right", yanchor="top",
    pad={"r": 10, "t": 40},
    buttons=[
        dict(label="▶ Odtwórz", method="animate",
             args=[None, dict(frame=dict(duration=40, redraw=True), fromcurrent=True, mode="immediate")]),
        dict(label="⏸ Pauza", method="animate",
             args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
    ]
)]

sliders = [dict(
    active=0, x=0.08, y=0, xanchor="left", yanchor="top", pad={"b": 10, "t": 40}, len=0.88,
    currentvalue=dict(prefix="Czas: ", font=dict(size=14), visible=True, xanchor="right"),
    steps=[dict(label=time_labels[i], method="animate",
                args=[[f"frame_{i}"], dict(frame=dict(duration=0, redraw=True), mode="immediate")]) for i in
           range(len(centers_lon))]
)]

fig.update_geos(
    projection_type="orthographic",
    projection_rotation=dict(lon=CAMERA_LON, lat=CAMERA_LAT, roll=0),
    showcoastlines=True, coastlinecolor="black", coastlinewidth=0.5,
    showland=True, landcolor="#FCD299", showocean=True, oceancolor="#94C1EC",
    showcountries=True, countrycolor="rgba(0,0,0,0.15)",
    lataxis=dict(showgrid=True, gridcolor="rgba(0, 0, 0, 0.15)", gridwidth=0.5, dtick=15),
    lonaxis=dict(showgrid=True, gridcolor="rgba(0, 0, 0, 0.15)", gridwidth=0.5, dtick=15)
)

fig.update_layout(
    title=dict(
        text="<b>Zaćmienie Słońca 12 Sierpnia 2026 – Wysoka Precyzja (JPL DE440)</b><br><span style='font-size:13px;'>Symulacja na 200 krokach czasowych z wykorzystaniem oficjalnych efemeryd NASA.</span>",
        x=0.5, y=0.96, xanchor='center'
    ),
    width=1000, height=1100, margin=dict(l=40, r=40, t=80, b=100),
    updatemenus=updatemenus, sliders=sliders,
    legend=dict(orientation="h", y=0.06, x=0.5, xanchor="center", bgcolor="white", bordercolor="gray", borderwidth=1)
)

output_filename = "zacmienie_2026_jpl_de440_x1000.html"
fig.write_html(output_filename)
print(f"\n[SUKCES] Wygenerowano plik: {output_filename}")
fig.show()
# Welcome

A clean, distraction-free markdown editor. Type on the left, see the rendered output on the right.

---

## Eclipse 2026

The Eclipse 2026 project consists of scripts written in Python 3.10 using astronomy libraries to calculate the course of the solar and lunar eclipse in August 2026.

## opis

The project began with calculations of the path of the solar eclipse on August 12, 2026. Python 3.10 and astronomy and other libraries were used to generate a JavaScript animation in the form of a web page. The project's output shows the path of the solar eclipse on Earth's surface on August 12, 2026, as well as the area of ​​the partial eclipse.

## Task Lists

- numpy
- plotly
- scipy
- astropy

## Links and Images

link to the page with the eclipse animation [inline links](https://astronomia.zagan.pl/pliki/zacmienie2026/zacmienie_2026_jpl_de440_x1000.html).

image from animation:

![Placeholder](https://github.com/jpatka/Solar-Eclipse-2026/blob/main/zacmienie12-08-2026_1.png?raw=true)



## Blockquotes

> The art of writing is the art of discovering what you believe.
>
> — Gustave Flaubert

## Code

Fenced code blocks support syntax highlighting:


```python
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
```

```javascript
<link rel="stylesheet"
href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">
<script language="javascript">
  function isInternetExplorer() {
    ua = navigator.userAgent;
    /* MSIE used to detect old browsers and Trident used to newer ones*/
    return ua.indexOf("MSIE ") > -1 || ua.indexOf("Trident/") > -1;
  }

  /* Define the Animation class */
  function Animation(frames, img_id, slider_id, interval, loop_select_id){
    this.img_id = img_id;
    this.slider_id = slider_id;
    this.loop_select_id = loop_select_id;
    this.interval = interval;
    this.current_frame = 0;
    this.direction = 0;
    this.timer = null;
    this.frames = new Array(frames.length);
```

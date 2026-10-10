from paraview.simple import *
import sys, os

case  = sys.argv[1]
out   = sys.argv[2]
W     = int(sys.argv[3])
H     = int(sys.argv[4])
title = sys.argv[5]
mode  = sys.argv[6]          # 'wide' -> barra horizontal abajo ; 'tall' -> barra vertical derecha

foam = os.path.join(case, 'case.foam')
open(foam, 'a').close()

r = OpenFOAMReader(registrationName='r', FileName=foam)
r.MeshRegions = ['internalMesh']
r.CellArrays = ['Ma', 'p', 'T', 'U', 'rho']
r.UpdatePipeline()
t = list(r.TimestepValues)[-1]

ref = Reflect(registrationName='ref', Input=r)
ref.Plane = 'Y Min'
ref.UpdatePipeline(t)

b = ref.GetDataInformation().GetBounds()
xmin, xmax, ymin, ymax = b[0], b[1], b[2], b[3]
cx, cy = 0.5*(xmin+xmax), 0.5*(ymin+ymax)
print('bounds x %.4f..%.4f  y %.4f..%.4f' % (xmin, xmax, ymin, ymax))

view = CreateRenderView()
view.ViewSize = [W, H]
view.OrientationAxesVisibility = 0
view.UseColorPaletteForBackground = 0
view.Background = [1.0, 1.0, 1.0]
view.InteractionMode = '2D'
view.CameraParallelProjection = 1
view.ViewTime = t

d = Show(ref, view)
d.Representation = 'Surface'
ColorBy(d, ('CELLS', 'Ma'))

lut = GetColorTransferFunction('Ma')
lut.ApplyPreset('Turbo', True)
d.RescaleTransferFunctionToDataRange(True, False)

bar = GetScalarBar(lut, view)
bar.Title = title
bar.ComponentTitle = ''
bar.TitleColor = [0, 0, 0]
bar.LabelColor = [0, 0, 0]
bar.TitleFontSize = 16
bar.LabelFontSize = 13
bar.WindowLocation = 'Any Location'

aspect = float(W) / float(H)

if mode == 'wide':
    # encuadre por ancho, barra horizontal en el espacio blanco de abajo
    half_w = 0.5 * (xmax - xmin) * 1.05
    scale = half_w / aspect
    bar.Orientation = 'Horizontal'
    bar.Position = [0.28, 0.06]
    bar.ScalarBarLength = 0.44
    bar.ScalarBarThickness = 13
else:
    # encuadre por alto, barra vertical a la derecha
    scale = 0.5 * (ymax - ymin) * 1.06
    bar.Orientation = 'Vertical'
    bar.Position = [0.855, 0.24]
    bar.ScalarBarLength = 0.52
    bar.ScalarBarThickness = 15

d.SetScalarBarVisibility(view, True)

# primer render: ParaView aplica su ResetCamera automatico
Render()

# ahora si se fija el encuadre, y se vuelve a renderizar
view.CameraFocalPoint = [cx, cy, 0.0]
view.CameraPosition   = [cx, cy, 10.0]
view.CameraViewUp     = [0.0, 1.0, 0.0]
view.CameraParallelScale = scale
Render()
print('parallelScale aplicado: %.4f' % view.CameraParallelScale)
SaveScreenshot(out, view, ImageResolution=[W, H], TransparentBackground=0)

rng = lut.RGBPoints
print('t=%.4f  rango Ma %.4f..%.4f  -> %s' % (t, rng[0], rng[-4], out))

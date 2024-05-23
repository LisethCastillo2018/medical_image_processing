
import SimpleITK as sitk


def registration(fixed_image_path, moving_image_path):

    fixed_image = sitk.ReadImage(fixed_image_path, sitk.sitkFloat32)
    moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)

    # Inicializar el registro con una transformación afín
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, 
        moving_image, 
        sitk.AffineTransform(fixed_image.GetDimension()),  
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # Configurar el método de registro
    registration_method = sitk.ImageRegistrationMethod()

    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    # Interpolador: Nearest Neighbor
    registration_method.SetInterpolator(sitk.sitkNearestNeighbor)

    # Optimizador: Funciones de gradiente descendiente
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10,
    )

    ## Introducción de un tercer sistema de coordenadas, el dominio de imagen virtual.
    ## Lo que permite tratar ambas imágenes (fixed y moving) de la misma manera
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Configuración para el marco de resolución múltiple.
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    # Se establece las transformaciones iniciales en movimiento y optimizadas.
    optimized_transform = sitk.Euler3DTransform()
    registration_method.SetMovingInitialTransform(initial_transform)
    registration_method.SetInitialTransform(optimized_transform, inPlace=False)

    # Es necesario componer las transformaciones después del registro.
    final_transform = sitk.CompositeTransform(
        [registration_method.Execute(fixed_image, moving_image), initial_transform]
    )

    # Aplicar la transformación a la imagen móvil
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(final_transform)

    # Imagen registrada
    registered_image = resampler.Execute(moving_image)

    # Guardar la imagen registrada
    sitk.WriteImage(registered_image, './store/imagen_registrada.nii')

    return registered_image

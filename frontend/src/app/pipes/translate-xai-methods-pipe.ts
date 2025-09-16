import { Pipe, PipeTransform } from '@angular/core';

@Pipe({
  name: 'translateXaiMethods',
})
export class TranslateXaiMethodsPipe implements PipeTransform {
  transform(value: unknown, ...args: unknown[]): unknown {
    if (value === 'gradient') {
      return 'Gradient';
    }
    if (value === 'gradient_squared') {
      return 'Gradient Squared';
    }
    if (value === 'deconvolution') {
      return 'Deconvolution';
    }
    if (value === 'guided_backpropagation') {
      return 'Guided Backpropagation';
    }
    if (value === 'input_gradient') {
      return 'Input x Gradient';
    }
    if (value === 'integrated_gradient') {
      return 'Integrated Gradient';
    }
    if (value === 'smooth_gradient') {
      return 'SmoothGrad';
    }
    if (value === 'smooth_gradient_square') {
      return 'SmoothGrad Squared';
    }
    if (value === 'var_gradient') {
      return 'VarGrad';
    }
    if (value === 'grad_cam') {
      return 'Grad-CAM';
    }
    if (value === 'grad_cam_plus_plus') {
      return 'Grad-CAM++';
    }
    if (value === 'group_cam') {
      return 'Group-CAM';
    }
    if (value === 'hires_cam') {
      return 'HiRes-CAM';
    }
    if (value === 'wgrad_cam') {
      return 'Weight Grad-CAM';
    }
    if (value === 'ablation_cam') {
      return 'Ablation-CAM';
    }
    if (value === 'xgrad_cam') {
      return 'XGrad-CAM';
    }
    if (value === 'lime') {
      return 'LIME';
    }
    if (value === 'kernel_shap') {
      return 'Kernel SHAP';
    }
    if (value === 'xrai') {
      return 'XRAI';
    }
    return '';
  }
}

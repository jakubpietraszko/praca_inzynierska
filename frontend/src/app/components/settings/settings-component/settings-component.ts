import { Component, inject, OnInit, effect } from '@angular/core';
import { CommonModule } from '@angular/common';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatSelectModule } from '@angular/material/select';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatCardModule } from '@angular/material/card';
import { FormsModule } from '@angular/forms';
import { Store } from '@ngrx/store';
import { toSignal } from '@angular/core/rxjs-interop';
import { AppState } from '../../../state/app.state';
import {
  ConfigModel,
  XAIMethodConfig,
} from '../../../state/config/config.model';
import * as ConfigActions from '../../../state/config/config.actions';
import * as ConfigSelectors from '../../../state/config/config.selectors';
import { TranslateXaiMethodsPipe } from '../../../pipes/translate-xai-methods-pipe';

@Component({
  selector: 'app-settings-component',
  imports: [
    CommonModule,
    MatFormFieldModule,
    MatSelectModule,
    MatInputModule,
    MatButtonModule,
    MatProgressSpinnerModule,
    MatExpansionModule,
    MatCardModule,
    FormsModule,
    TranslateXaiMethodsPipe,
  ],
  templateUrl: './settings-component.html',
  styleUrl: './settings-component.css',
})
export class SettingsComponent implements OnInit {
  private store = inject(Store<AppState>);

  config = toSignal(this.store.select(ConfigSelectors.selectConfig));
  loading = toSignal(this.store.select(ConfigSelectors.selectConfigLoading), {
    initialValue: false,
  });
  methods = toSignal(this.store.select(ConfigSelectors.selectMethods), {
    initialValue: {},
  });
  methodNames = toSignal(this.store.select(ConfigSelectors.selectMethodNames), {
    initialValue: [],
  });

  currentConfig: { [methodName: string]: XAIMethodConfig } = {};

  aggregationOptions = ['max', 'sum', 'max'];
  segmentationOptions = ['seed', 'seed'];

  constructor() {
    effect(() => {
      const methods = this.methods();
      if (methods && Object.keys(methods).length > 0) {
        this.currentConfig = { ...methods };
      } else {
        this.initializeDefaultConfig();
      }
    });
  }

  ngOnInit() {
    this.store.dispatch(ConfigActions.loadConfig());
  }

  initializeDefaultConfig() {
    const defaultMethods = [
      'grad_cam',
      'grad_cam_plus_plus',
      'group_cam',
      'hires_cam',
      'wgrad_cam',
      'ablation_cam',
      'xgrad_cam',
      'smooth_grad',
      'smooth_grad_squared',
      'var_grad',
      'gradient',
      'gradient_squared',
      'deconvolution',
      'guided_backpropagation',
      'input_gradient',
      'intergrated_gradient',
      'xrai',
      'kernel_shap',
      'lime',
    ];

    this.currentConfig = {};
    defaultMethods.forEach((methodName) => {
      this.currentConfig[methodName] = this.getDefaultMethodConfig(methodName);
    });
  }

  getDefaultMethodConfig(methodName: string): XAIMethodConfig {
    const defaultConfigs: { [key: string]: XAIMethodConfig } = {
      grad_cam: { epsilon: 1e-6 },
      grad_cam_plus_plus: { epsilon: 1e-6 },
      group_cam: {
        epsilon: 1e-6,
        group_size: 16,
        k_size: 4,
        sigma: 0.5,
      },
      hires_cam: { epsilon: 1e-6 },
      wgrad_cam: { epsilon: 1e-6 },
      ablation_cam: { epsilon: 1e-6 },
      xgrad_cam: { epsilon: 1e-6 },
      activation_cam: { epsilon: 1e-6 },
      smooth_grad: {
        epsilon: 1e-6,
        n_samples: 50,
        noise_level: 0.1,
        aggregation: 'max',
      },
      smooth_grad_squared: {
        epsilon: 1e-6,
        n_samples: 50,
        noise_level: 0.1,
        aggregation: 'max',
      },
      var_grad: {
        epsilon: 1e-6,
        n_samples: 50,
        noise_level: 0.1,
        aggregation: 'max',
      },
      gradient: { epsilon: 1e-6, aggregation: 'max' },
      gradient_squared: { epsilon: 1e-6, aggregation: 'max' },
      deconvolution: { epsilon: 1e-6, aggregation: 'max' },
      guided_backpropagation: { epsilon: 1e-6, aggregation: 'max' },
      input_gradient: { epsilon: 1e-6, aggregation: 'max' },
      intergrated_gradient: {
        epsilon: 1e-6,
        baseline: 0,
        steps: 50,
        aggregation: 'max',
      },
      xrai: {
        epsilon: 1e-6,
        aggregation: 'max',
        baseline: 0,
        super_pixels: 30,
        segmentation: 'seed',
      },
      kernel_shap: {
        epsilon: 1e-6,
        n_samples: 50,
        aggregation: 'max',
        baseline: 0,
        super_pixels: 30,
        segmentation: 'seed',
      },
      lime: {
        epsilon: 1e-6,
        n_samples: 50,
        aggregation: 'max',
        baseline: 0,
        super_pixels: 30,
        segmentation: 'seed',
      },
    };

    return defaultConfigs[methodName] || { epsilon: 1e-6, aggregation: 'max' };
  }

  onMethodConfigChange(
    methodName: string,
    field: keyof XAIMethodConfig,
    value: any
  ) {
    console.log('Updating config:', methodName, field, value); // Debug log

    if (!this.currentConfig[methodName]) {
      this.currentConfig[methodName] = this.getDefaultMethodConfig(methodName);
    }

    if (
      [
        'epsilon',
        'group_size',
        'k_size',
        'sigma',
        'n_samples',
        'noise_level',
        'steps',
        'baseline',
        'super_pixels',
      ].includes(field)
    ) {
      const numValue = parseFloat(value);
      if (isNaN(numValue)) return;
      value = numValue;
    }

    this.currentConfig[methodName] = {
      ...this.currentConfig[methodName],
      [field]: value,
    };

    this.store.dispatch(
      ConfigActions.updateMethodConfig({
        methodName,
        methodConfig: { [field]: value },
      })
    );
  }

  onSaveConfig() {
    const config: ConfigModel = { methods: this.currentConfig };
    this.store.dispatch(ConfigActions.saveConfig({ config }));
  }

  onResetConfig() {
    this.store.dispatch(ConfigActions.resetConfig());
  }

  getMethodConfig(methodName: string): XAIMethodConfig {
    if (this.currentConfig[methodName]) {
      return this.currentConfig[methodName];
    }
    return this.getDefaultMethodConfig(methodName);
  }

  hasField(methodName: string, field: string): boolean {
    return this.getDefaultFields(methodName).includes(field);
  }

  getDefaultFields(methodName: string): string[] {
    const fieldMap: { [key: string]: string[] } = {
      grad_cam: ['epsilon'],
      grad_cam_plus_plus: ['epsilon'],
      group_cam: ['epsilon', 'group_size', 'k_size', 'sigma'],
      hires_cam: ['epsilon'],
      wgrad_cam: ['epsilon'],
      ablation_cam: ['epsilon'],
      xgrad_cam: ['epsilon'],
      smooth_grad: ['epsilon', 'n_samples', 'noise_level', 'aggregation'],
      smooth_grad_squared: [
        'epsilon',
        'n_samples',
        'noise_level',
        'aggregation',
      ],
      var_grad: ['epsilon', 'n_samples', 'noise_level', 'aggregation'],
      gradient: ['epsilon', 'aggregation'],
      gradient_squared: ['epsilon', 'aggregation'],
      deconvolution: ['epsilon', 'aggregation'],
      guided_backpropagation: ['epsilon', 'aggregation'],
      input_gradient: ['epsilon', 'aggregation'],
      intergrated_gradient: ['epsilon', 'baseline', 'steps', 'aggregation'],
      xrai: [
        'epsilon',
        'aggregation',
        'baseline',
        'super_pixels',
        'segmentation',
      ],
      kernel_shap: [
        'epsilon',
        'n_samples',
        'aggregation',
        'baseline',
        'super_pixels',
        'segmentation',
      ],
      lime: [
        'epsilon',
        'n_samples',
        'aggregation',
        'baseline',
        'super_pixels',
        'segmentation',
      ],
    };

    return fieldMap[methodName] || ['epsilon', 'aggregation'];
  }

  getMethodDisplayName(methodName: string): string {
    return methodName
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (l) => l.toUpperCase());
  }

  getAllMethodNames(): string[] {
    const loadedMethods = this.methodNames() || [];
    const defaultMethods = [
      'grad_cam',
      'grad_cam_plus_plus',
      'group_cam',
      'hires_cam',
      'wgrad_cam',
      'ablation_cam',
      'xgrad_cam',
      'activation_cam',
      'smooth_grad',
      'smooth_grad_squared',
      'var_grad',
      'gradient',
      'gradient_squared',
      'deconvolution',
      'guided_backpropagation',
      'input_gradient',
      'intergrated_gradient',
      'xrai',
      'kernel_shap',
      'lime',
    ];

    return loadedMethods.length > 0 ? loadedMethods : defaultMethods;
  }
}

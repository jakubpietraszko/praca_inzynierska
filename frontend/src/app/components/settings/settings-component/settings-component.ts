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
import { ConfigStore } from '../../../stores/config.store';
import {
  ConfigModel,
  XAIMethodConfig,
} from '../../../state/config/config.model';

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
  ],
  templateUrl: './settings-component.html',
  styleUrl: './settings-component.css',
})
export class SettingsComponent implements OnInit {
  private configStore = inject(ConfigStore);

  config = this.configStore.config;
  loading = this.configStore.loading;
  methods = this.configStore.methods;
  methodNames = this.configStore.methodNames;
  error = this.configStore.error;

  currentConfig: { [methodName: string]: XAIMethodConfig } = {};

  aggregationOptions = ['max', 'sum', 'mean'];
  segmentationOptions = ['seed', 'slic'];

  constructor() {
    effect(() => {
      const methods = this.methods();
      const loading = this.loading();

      if (!loading && methods && Object.keys(methods).length > 0) {
        this.currentConfig = JSON.parse(JSON.stringify(methods));
      }
    });
  }

  ngOnInit() {
    this.configStore.loadConfig();
  }

  getDefaultMethodConfig(methodName: string): XAIMethodConfig {
    return this.getMethodConfig(methodName);
  }

  onMethodConfigChange(
    methodName: string,
    field: keyof XAIMethodConfig,
    value: any
  ) {
    const currentMethodConfig =
      this.currentConfig[methodName] || this.getMethodConfig(methodName);

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
        'gamma',
        'alpha',
        'beta',
      ].includes(field)
    ) {
      const numValue = parseFloat(value);
      if (isNaN(numValue)) return;
      value = numValue;
    }

    if (field === 'k_size') {
      const currentValue = currentMethodConfig?.k_size || 5;
      const minValue = Math.max(1, currentValue - 2);
      const maxValue = currentValue + 2;

      if (value % 2 === 0) {
        value = value + 1;
      }

      if (value < minValue) {
        value = minValue % 2 === 0 ? minValue + 1 : minValue;
      } else if (value > maxValue) {
        value = maxValue % 2 === 0 ? maxValue - 1 : maxValue;
      }

      if (value % 2 === 0) {
        value = value - 1;
      }

      if (value < 1) {
        value = 1;
      }
    }

    this.currentConfig = {
      ...this.currentConfig,
      [methodName]: {
        ...currentMethodConfig,
        [field]: value,
      },
    };

    this.configStore.updateMethodConfig(methodName, { [field]: value });
  }

  onSaveConfig() {
    const config: ConfigModel = { methods: this.currentConfig };
    this.configStore.saveConfig(config);
  }

  onResetConfig() {
    this.configStore.resetConfig();
  }

  getMethodConfig(methodName: string): XAIMethodConfig {
    if (this.currentConfig[methodName]) {
      return this.currentConfig[methodName];
    }

    const storeConfig = this.configStore.getMethodConfig(methodName);
    if (storeConfig) {
      return storeConfig;
    }

    return { epsilon: 1e-6 };
  }

  hasField(methodName: string, field: string): boolean {
    return this.getDefaultFields(methodName).includes(field);
  }

  getDefaultFields(methodName: string): string[] {
    const methodConfig = this.getMethodConfig(methodName);
    if (methodConfig) {
      return Object.keys(methodConfig).filter((key) => {
        const value = (methodConfig as any)[key];
        return value !== null && value !== undefined;
      });
    }

    return ['epsilon'];
  }

  getMethodDisplayName(methodName: string): string {
    return methodName
      .replace(/_/g, ' ')
      .replace(/\b\w/g, (l) => l.toUpperCase());
  }

  getAllMethodNames(): string[] {
    const methodNames = this.methodNames() || [];

    return methodNames;
  }
}

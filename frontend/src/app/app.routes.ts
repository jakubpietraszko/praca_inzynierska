import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    redirectTo: 'models',
    pathMatch: 'full',
  },
  {
    path: 'settings',
    loadComponent: () => import('./components/settings-page/settings').then(m => m.Settings),
  },
  {
    path: 'datasets',
    loadComponent: () => import('./components/datasets-page/datasets').then(m => m.Datasets),
  },
  {
    path: 'models',
    loadComponent: () => import('./components/models-page/models').then(m => m.Models),
  },
  {
    path: 'statistics',
    loadComponent: () => import('./components/statistics-page/statistics').then(m => m.Statistics),
  },
  {
    path: 'generator',
    loadComponent: () => import('./components/generator-page/generator').then(m => m.Generator),
  },
  {
    path: '**',
    redirectTo: 'models',
    pathMatch: 'full',
  },
];

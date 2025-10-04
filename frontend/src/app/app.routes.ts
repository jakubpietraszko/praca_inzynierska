import { Routes } from '@angular/router';
import { Generator } from './components/generator-page/generator';
import { Statistics } from './components/statistics-page/statistics';
import { Datasets } from './components/datasets-page/datasets';
import { Settings } from './components/settings-page/settings';
import { Home } from './components/home-page/home';
import { Models } from './components/models-page/models';

export const routes: Routes = [
  {
    path: 'home',
    component: Home,
    pathMatch: 'full',
  },
  {
    path: 'settings',
    component: Settings,
    pathMatch: 'full',
  },
  {
    path: 'datasets',
    component: Datasets,
    pathMatch: 'full',
  },
  {
    path: 'models',
    component: Models,
    pathMatch: 'full',
  },
  {
    path: 'statistics',
    component: Statistics,
    pathMatch: 'full',
  },
  {
    path: 'generator',
    component: Generator,
    pathMatch: 'full',
  },
  {
    path: '**',
    redirectTo: 'home',
    pathMatch: 'full',
  },
];

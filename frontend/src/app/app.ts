import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { ImageComponent } from './components-test/image-component/image-component';
import { NavbarComponent } from './components/navbar/navbar-component/navbar-component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, ImageComponent, NavbarComponent],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly title = signal('frontend');
}

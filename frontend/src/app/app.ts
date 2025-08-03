import { Component, signal } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { ImageComponent } from './components-test/image-component/image-component';

@Component({
  selector: 'app-root',
  imports: [RouterOutlet, ImageComponent],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  protected readonly title = signal('frontend');
}

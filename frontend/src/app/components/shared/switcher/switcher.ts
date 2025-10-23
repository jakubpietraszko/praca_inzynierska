import {
  ChangeDetectionStrategy,
  Component,
  computed,
  input,
  model,
} from '@angular/core';
import { MatButtonModule } from '@angular/material/button';
import { MatIconModule } from '@angular/material/icon';

@Component({
  selector: 'app-switcher',
  imports: [MatIconModule, MatButtonModule],
  templateUrl: './switcher.html',
  styleUrl: './switcher.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class Switcher {
  items = input.required<string[]>();
  selectedItem = model.required<string>();

  protected currentIndex = computed(() => {
    const items = this.items();
    const selectedItem = this.selectedItem();
    return items.indexOf(selectedItem);
  });

  protected itemIsFirst = computed(() => {
    return this.currentIndex() === 0;
  });

  protected itemIsLast = computed(() => {
    return this.currentIndex() === this.items().length - 1;
  });

  protected selectPrevious() {
    const index = this.currentIndex();
    if (!this.itemIsFirst()) {
      const previousItem = this.items()[index - 1];
      this.selectedItem.set(previousItem);
    }
  }

  protected selectNext() {
    const index = this.currentIndex();
    if (!this.itemIsLast()) {
      const nextItem = this.items()[index + 1];
      this.selectedItem.set(nextItem);
    }
  }
}

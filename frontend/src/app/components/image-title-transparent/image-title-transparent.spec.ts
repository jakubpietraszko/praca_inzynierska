import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ImageTitleTransparent } from './image-title-transparent';

describe('ImageTitleTransparent', () => {
  let component: ImageTitleTransparent;
  let fixture: ComponentFixture<ImageTitleTransparent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [ImageTitleTransparent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ImageTitleTransparent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

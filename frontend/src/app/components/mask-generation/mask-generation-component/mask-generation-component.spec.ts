import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MaskGenerationComponent } from './mask-generation-component';

describe('MaskGenerationComponent', () => {
  let component: MaskGenerationComponent;
  let fixture: ComponentFixture<MaskGenerationComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [MaskGenerationComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(MaskGenerationComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});

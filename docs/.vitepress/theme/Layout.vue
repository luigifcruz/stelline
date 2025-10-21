<script setup lang="ts">
import DefaultTheme from 'vitepress/theme'

const { Layout } = DefaultTheme
</script>

<template>
  <Layout>
    <template #layout-top>
      <div class="dev-indicator">
        <svg class="dev-indicator-icon" xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
          <line x1="12" y1="9" x2="12" y2="13"/>
          <line x1="12" y1="17" x2="12.01" y2="17"/>
        </svg>
        <span class="dev-indicator-text">Docs in Development</span>
      </div>
    </template>
  </Layout>
</template>

<style scoped>
.dev-indicator {
  position: fixed;
  bottom: 24px;
  right: 24px;
  display: flex;
  align-items: center;
  gap: 0.5rem;
  padding: 0.75rem 1rem;
  background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%);
  color: #78350f;
  border-radius: 9999px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15), 0 0 0 1px rgba(0, 0, 0, 0.1);
  z-index: 1000;
  font-size: 0.875rem;
  font-weight: 600;
  backdrop-filter: blur(8px);
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.dev-indicator:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 16px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(0, 0, 0, 0.1);
}

.dev-indicator-icon {
  flex-shrink: 0;
  color: #78350f;
}

.dev-indicator-text {
  white-space: nowrap;
  color: #78350f;
}

@media (max-width: 768px) {
  .dev-indicator {
    bottom: 16px;
    right: 16px;
    padding: 0.625rem 0.875rem;
    font-size: 0.8125rem;
  }

  .dev-indicator-icon {
    width: 14px;
    height: 14px;
  }
}

@media (max-width: 480px) {
  .dev-indicator {
    bottom: 12px;
    right: 12px;
    padding: 0.5rem 0.75rem;
    font-size: 0.75rem;
  }
}
</style>

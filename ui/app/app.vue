<script setup lang="ts">
useHead({
  link: [
    {
      rel: "icon",
      type: "image/png",
      sizes: "32x32",
      href: "/favicon-32x32.png"
    },
    {
      rel: "icon",
      type: "image/png",
      sizes: "16x16",
      href: "/favicon-16x16.png"
    },
    {
      rel: "apple-touch-icon",
      sizes: "180x180",
      href: "/apple-touch-icon.png"
    },
    {
      rel: "manifest",
      href: "/site.webmanifest"
    }
  ]
})
</script>

<template>
  <div>
    <NuxtRouteAnnouncer />
    <UApp>
      <NuxtLayout>
        <NuxtPage />
      </NuxtLayout>
    </UApp>
  </div>
</template>

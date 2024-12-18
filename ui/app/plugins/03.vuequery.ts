import type {
  DehydratedState,
  VueQueryPluginOptions,
} from "@tanstack/vue-query";
import {
  VueQueryPlugin,
  QueryClient,
  hydrate,
  dehydrate,
  MutationCache,
} from "@tanstack/vue-query";
// Nuxt 3 app aliases
import { defineNuxtPlugin, useState } from "#imports";
import type { NuxtError } from "#app";

export default defineNuxtPlugin((nuxt) => {
  const vueQueryState = useState<DehydratedState | null>("vue-query");

  // Modify your Vue Query global settings here
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { staleTime: 5000, refetchIntervalInBackground: true },
    },
    mutationCache: new MutationCache({
      onError: (error) => {
        const toast = useToast();
        toast.add({
          title: error.data?.detail?.description || "Something went wrong",
          color: "error",
          icon: "i-lucide-x-circle",
        });
      },
    }),
  });
  const options: VueQueryPluginOptions = { queryClient };

  nuxt.vueApp.use(VueQueryPlugin, options);

  if (import.meta.server) {
    nuxt.hooks.hook("app:rendered", () => {
      vueQueryState.value = dehydrate(queryClient);
    });
  }

  if (import.meta.client) {
    hydrate(queryClient, vueQueryState.value);
  }
});

declare module "@tanstack/vue-query" {
  interface Register {
    defaultError: NuxtError<{ detail: { description: string } }>;
  }
}

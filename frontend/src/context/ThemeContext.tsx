import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import type {
  ReactNode,
} from "react";

export type Theme = "light" | "dark";

interface ThemeContextValue {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
}

const ThemeContext =
  createContext<ThemeContextValue | undefined>(
    undefined,
  );

const THEME_STORAGE_KEY =
  "smart-research-theme";

function getInitialTheme(): Theme {
  const stored =
    window.localStorage.getItem(
      THEME_STORAGE_KEY,
    );

  if (
    stored === "light" ||
    stored === "dark"
  ) {
    return stored;
  }

  return window.matchMedia?.(
    "(prefers-color-scheme: dark)",
  ).matches
    ? "dark"
    : "light";
}

export function ThemeProvider({
  children,
}: {
  children: ReactNode;
}) {
  const [theme, setThemeState] =
    useState<Theme>(getInitialTheme);

  useEffect(() => {
    const root = document.documentElement;

    root.setAttribute(
      "data-theme",
      theme,
    );

    root.style.colorScheme = theme;

    window.localStorage.setItem(
      THEME_STORAGE_KEY,
      theme,
    );
  }, [theme]);

  const value = useMemo(
    () => ({
      theme,

      toggleTheme: () => {
        setThemeState((current) =>
          current === "light"
            ? "dark"
            : "light",
        );
      },

      setTheme: (nextTheme: Theme) => {
        setThemeState(nextTheme);
      },
    }),
    [theme],
  );

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
}

export function useTheme() {
  const context =
    useContext(ThemeContext);

  if (!context) {
    throw new Error(
      "useTheme must be used inside ThemeProvider.",
    );
  }

  return context;
}
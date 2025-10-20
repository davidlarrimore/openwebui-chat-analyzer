"use client";

import * as React from "react";
import { ToastActionElement, Toast, ToastClose, ToastDescription, ToastProvider, ToastTitle, ToastViewport } from "./toast";

const TOAST_LIMIT = 5;

type ToasterToast = {
  id: string;
  title?: React.ReactNode;
  description?: React.ReactNode;
  action?: ToastActionElement;
  duration?: number;
  variant?: "default" | "destructive";
};

type ToasterState = {
  toasts: ToasterToast[];
};

type Action =
  | { type: "ADD_TOAST"; toast: ToasterToast }
  | { type: "DISMISS_TOAST"; toastId?: ToasterToast["id"] }
  | { type: "REMOVE_TOAST"; toastId?: ToasterToast["id"] };

const toastTimeouts = new Map<string, ReturnType<typeof setTimeout>>();

const toastReducer = (state: ToasterState, action: Action): ToasterState => {
  switch (action.type) {
    case "ADD_TOAST": {
      return {
        ...state,
        toasts: [action.toast, ...state.toasts].slice(0, TOAST_LIMIT)
      };
    }
    case "DISMISS_TOAST":
      return {
        ...state,
        toasts: state.toasts.map((toast) =>
          toast.id === action.toastId ? { ...toast, dismissed: true } : toast
        )
      };
    case "REMOVE_TOAST": {
      if (action.toastId) {
        return {
          ...state,
          toasts: state.toasts.filter((toast) => toast.id !== action.toastId)
        };
      }
      return { ...state, toasts: [] };
    }
    default:
      return state;
  }
};

const listeners = new Set<(state: ToasterState) => void>();

let memoryState: ToasterState = { toasts: [] };

function dispatch(action: Action) {
  memoryState = toastReducer(memoryState, action);
  listeners.forEach((listener) => {
    listener(memoryState);
  });
}

type ToastOptions = Omit<ToasterToast, "id">;

function generateId() {
  return Math.random().toString(36).slice(2, 10);
}

export function toast({ title, description, action, duration, variant }: ToastOptions) {
  const id = generateId();
  const dismiss = () => dispatch({ type: "DISMISS_TOAST", toastId: id });
  const remove = () => dispatch({ type: "REMOVE_TOAST", toastId: id });

  dispatch({
    type: "ADD_TOAST",
    toast: {
      id,
      title,
      description,
      action,
      duration: duration ?? 3000,
      variant
    }
  });

  if (duration !== undefined) {
    const timeout = setTimeout(() => dismiss(), duration);
    toastTimeouts.set(id, timeout);
  }

  return {
    id,
    dismiss,
    remove
  };
}

export function useToast() {
  const [state, setState] = React.useState<ToasterState>(memoryState);

  React.useEffect(() => {
    listeners.add(setState);
    return () => {
      listeners.delete(setState);
    };
  }, []);

  return {
    ...state,
    toast,
    dismiss: (toastId?: string) => dispatch({ type: "DISMISS_TOAST", toastId })
  };
}

export function Toaster() {
  const { toasts } = useToast();

  return (
    <ToastProvider>
      {toasts.map(({ id, title, description, action, duration, variant }) => (
        <Toast key={id} duration={duration} className={variant === "destructive" ? "border-destructive bg-destructive text-destructive-foreground" : ""}>
          <div className="grid gap-1">
            {title && <ToastTitle>{title}</ToastTitle>}
            {description && <ToastDescription>{description}</ToastDescription>}
          </div>
          {action}
          <ToastClose />
        </Toast>
      ))}
      <ToastViewport />
    </ToastProvider>
  );
}
